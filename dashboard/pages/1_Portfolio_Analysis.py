# dashboard/pages/1_Portfolio_Analysis.py

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.stats import t as tdist

from utils.loaders import load_data
from risk.student_t import compute_student_t_stats
from risk.portfolio_t import estimate_portfolio_df, portfolio_t_var_es
from risk.garch import garch_fit
from risk.portfolio_garch import portfolio_garch_var_es
from risk.historical import historical_var_es
from risk.copulas import mc_portfolio_pnl, var_cvar


st.title("📌 Portfolio Analysis – What’s Driving Your Risk?")

# --------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------
df_ret, w = load_data()
rng = default_rng(42)
port_ret = df_ret @ w
notional = 1_000_000

st.markdown(
    f"""
- **Assets:** {len(w[w > 0])}  
- **History length:** {len(df_ret)} daily observations  
- **Notional for risk numbers:** €{notional:,.0f}
"""
)

# --------------------------------------------------------------
# 2. Per-asset static Student-t risk
# --------------------------------------------------------------
st.subheader("Per-asset risk – static Student-t")

stats_t = compute_student_t_stats(df_ret, rng)

# consistent column names
es95_col   = "ES95"
var95_col  = "VaR95"
mean_col   = "mean"
std_col    = "std"
df_col     = "df"

# Sort: highest ES (least negative → safest) down to most negative (riskiest)
stats_t_sorted = stats_t.sort_values(es95_col, ascending=False)
best_5_t  = stats_t_sorted.head(5)
worst_5_t = stats_t_sorted.tail(5)

display_cols = [var95_col, es95_col]
c1, c2 = st.columns(2)

with c1:
    st.markdown("#### Best 5 assets by 95% ES (static Student-t)")
    st.dataframe(
        best_5_t[display_cols].applymap(lambda x: f"{x:.2%}")
    )

with c2:
    st.markdown("#### Worst 5 assets by 95% ES (static Student-t)")
    st.dataframe(
        worst_5_t[display_cols].applymap(lambda x: f"{x:.2%}")
    )

st.markdown(
    """
**Interpretation**

- Each asset’s returns are modelled as i.i.d. Student-t with **constant** volatility.  
- ES 95% is the *average loss* in the worst 5% of days *for that asset alone*.  
- The left table shows the **most stable names**; the right shows the **largest static
  tail-risk contributors** under this simple model.
"""
)

# --------------------------------------------------------------
# 3. Per-asset conditional GARCH-t risk
# --------------------------------------------------------------
st.subheader("Per-asset risk – conditional GARCH-t")

# Fit GARCH(1,1) with Student-t innovations per asset:
# each entry: (nu_garch, mu_garch, sigma_garch)
garch_out = df_ret.apply(garch_fit)
dfs_g = garch_out.apply(lambda x: x[0])
mu_g  = garch_out.apply(lambda x: x[1])
sig_g = garch_out.apply(lambda x: x[2])

garch_stats = pd.DataFrame(
    {
        "df_garch":   dfs_g,
        "mean_garch": mu_g,
        "std_garch":  sig_g,
    }
)

def es_factor(alpha, nu):
    """
    ES multiplier for a left-tail Student-t(ν) at probability alpha.
    Returns a **negative** number (loss).
    """
    q   = tdist.ppf(alpha, df=nu)
    pdf = tdist.pdf(q, df=nu)
    return -((nu + q**2) / (nu - 1)) * (pdf / alpha)

# 1-day conditional VaR / ES at 95% (signed returns: negative = loss)
garch_stats["VaR95"] = (
    tdist.ppf(0.05, df=garch_stats["df_garch"]) * garch_stats["std_garch"]
    + garch_stats["mean_garch"]
)
garch_stats["ES95"] = (
    es_factor(0.05, garch_stats["df_garch"]) * garch_stats["std_garch"]
    + garch_stats["mean_garch"]
)

# Sort: highest ES (least negative → safest) down to most negative (riskiest)
garch_sorted = garch_stats.sort_values("ES95", ascending=False)
best_5_g  = garch_sorted.head(5)
worst_5_g = garch_sorted.tail(5)

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Best 5 assets by 95% ES (GARCH-t)")
    st.dataframe(
        best_5_g[["VaR95", "ES95"]].applymap(lambda x: f"{x:.2%}")
    )

with c2:
    st.markdown("#### Worst 5 assets by 95% ES (GARCH-t)")
    st.dataframe(
        worst_5_g[["VaR95", "ES95"]].applymap(lambda x: f"{x:.2%}")
    )

st.markdown(
    """
**Interpretation**

- Volatility is now **time-varying**, estimated via a GARCH(1,1) with Student-t shocks.  
- **VaR 95%** is the 1-day loss exceeded in only 5% of days for that asset under the
  current conditional volatility.  
- **ES 95%** is the *average loss* given that you are already in that worst 5% tail.  
- Comparing these rankings with the static Student-t section shows which names become
  **more or less risky in the current volatility regime**.
"""
)

# --------------------------------------------------------------
# 4. Portfolio-level VaR/ES – model comparison
# --------------------------------------------------------------
st.subheader("Portfolio-level VaR / ES – model comparison")

# 4.1 Static Student-t (variance–covariance style)
nu_p = estimate_portfolio_df(port_ret, rng)
VaR95_t, ES95_t, VaR99_t, ES99_t = portfolio_t_var_es(port_ret, nu_p)

# 4.2 GARCH-t portfolio
# (reuse garch_out above)
corr  = df_ret.corr().values
VaR95_g, ES95_g, VaR99_g, ES99_g = portfolio_garch_var_es(
    w.values, mu_g.values, sig_g.values, corr, dfs_g.values
)

# 4.3 Historical
hist95, hist_es95 = historical_var_es(port_ret, 0.05)
hist99, hist_es99 = historical_var_es(port_ret, 0.01)

# 4.4 Monte Carlo t-copula
pnl_1d = mc_portfolio_pnl(
    df_ret, w,
    notional=notional,
    n_sims=100_000,
    horizon_days=1,
    lam=0.94,
    nu_copula=5,
    df_marg=5,
)
var95_mc, es95_mc = var_cvar(pnl_1d, alpha=0.95)
var99_mc, es99_mc = var_cvar(pnl_1d, alpha=0.99)

# Put everything in a table (per-unit and €)
summary = pd.DataFrame({
    "Model": [
        "Static Student-t",
        "GARCH-t (conditional)",
        "Historical",
        "Monte Carlo t-copula",
    ],
    "VaR 95% (rel)": [VaR95_t, VaR95_g, hist95, var95_mc / notional],
    "ES 95% (rel)":  [ES95_t, ES95_g, hist_es95, es95_mc / notional],
    "VaR 99% (rel)": [VaR99_t, VaR99_g, hist99, var99_mc / notional],
    "ES 99% (rel)":  [ES99_t, ES99_g, hist_es99, es99_mc / notional],
})

summary_eur = summary.copy()
for col in ["VaR 95% (rel)", "ES 95% (rel)", "VaR 99% (rel)", "ES 99% (rel)"]:
    summary_eur[col] = summary[col] * notional

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Relative (fraction of portfolio value)")
    st.dataframe(summary.set_index("Model").applymap(lambda x: f"{x:.2%}"))
with c2:
    st.markdown("#### In euros (1-day horizon, notional €1,000,000)")
    st.dataframe(summary_eur.set_index("Model").applymap(lambda x: f"{x:,.0f} €"))

st.markdown(
    """
**What’s the difference between the models?**

- **Static Student-t**  
  - Uses a single covariance matrix and degrees of freedom estimated from history.  
  - Assumes *constant* volatility and correlation.  
  - Fast and simple, but may miss clustering of volatility.

- **GARCH-t**  
  - Volatility per asset is **time-varying** (GARCH(1,1)), correlations still static.  
  - Captures volatility clustering and fat tails without simulating full paths.

- **Historical VaR**  
  - Purely non-parametric: uses the empirical distribution of past portfolio returns.  
  - No model assumptions, but heavily dependent on the sample window.

- **Monte Carlo t-copula**  
  - Uses an **EWMA covariance** + **t-copula** to simulate 100k correlated scenarios.  
  - Captures tail dependence and non-Gaussian behaviour; gives both VaR and CVaR.  

Use this page to see **which assets** and **which modelling choices** drive the risk you
see in the later pages (MC distributions, stress tests, backtesting).
"""
)
