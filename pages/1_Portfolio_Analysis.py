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
import time

from utils.loaders import load_data
from risk.student_t import compute_student_t_stats
from risk.portfolio_t import estimate_portfolio_df, portfolio_t_var_es
from risk.garch import garch_fit
from risk.portfolio_garch import portfolio_garch_var_es
from risk.historical import historical_var_es
from risk.copulas import mc_portfolio_pnl, var_cvar

st.title("📌 Portfolio Analysis – What’s Driving Your Risk?")
st.markdown("### 🔧 DEBUG VERSION WITH INLINE TIMING")

# ==========================================================
# 1. Load data (timed)
# ==========================================================
t0 = time.perf_counter()
df_ret, w = load_data()
t1 = time.perf_counter()
st.markdown(f"**⏱ load_data:** {t1 - t0:.3f} s")

port_ret = df_ret @ w
notional = 1_000_000

# For GARCH: restrict to non-zero-weight assets & last 250 days
N_GARCH_DAYS = 250
active_mask = w > 0
active_cols = df_ret.columns[active_mask]          # <— only assets with w>0
df_ret_garch = df_ret[active_cols].iloc[-N_GARCH_DAYS:]

st.markdown(
    f"""
- **Active assets (w > 0):** {active_mask.sum()}  
- **Full history length:** {len(df_ret)}  
- **GARCH window:** last {len(df_ret_garch)} days  
- **Notional:** €{notional:,.0f}
"""
)

# ==========================================================
# 2. Per-asset static Student-t risk (timed)
# ==========================================================
st.subheader("Per-asset risk – static Student-t")

t0 = time.perf_counter()
rng = default_rng(42)

# 🔴 OLD (includes zero-weight assets)
# stats_t = compute_student_t_stats(df_ret, rng)

# ✅ NEW: only active assets
stats_t = compute_student_t_stats(df_ret[active_cols], rng)

t1 = time.perf_counter()
st.markdown(f"**⏱ compute_student_t_stats:** {t1 - t0:.3f} s")

es95_col = "ES95"
var95_col = "VaR95"
display_cols = [var95_col, es95_col]

stats_sorted = stats_t.sort_values(es95_col)

worst_5_t = stats_sorted.head(5)   # worst = most negative ES
best_5_t  = stats_sorted.tail(5)   # best  = least negative ES

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Best 5 (lowest risk)")
    st.dataframe(best_5_t[display_cols].applymap(lambda x: f"{x:.2%}"))

with c2:
    st.markdown("#### Worst 5 (highest risk)")
    st.dataframe(worst_5_t[display_cols].applymap(lambda x: f"{x:.2%}"))

st.markdown(
"""
**Interpretation**

- This fit assumes each asset’s returns are i.i.d. Student-t with **constant** volatility.  
- ES 95% is the *average loss* in the worst 5% of days *for that asset alone*.  
- Here you can clearly see which names are driving the **tail risk** only among **current holdings**.
"""
)

# ==========================================================
# 3. Per-asset conditional GARCH-t risk (timed)
# ==========================================================
st.subheader("Per-asset risk – conditional GARCH-t")
st.caption(
    f"GARCH computed only on active assets and last {N_GARCH_DAYS} days."
)

t0 = time.perf_counter()
garch_out = df_ret_garch.apply(garch_fit)  # each: (nu_g, mu_g, sigma_g)
t1 = time.perf_counter()
st.markdown(f"**⏱ garch_fit over assets:** {t1 - t0:.3f} s")

dfs_g = garch_out.apply(lambda x: x[0])
mu_g  = garch_out.apply(lambda x: x[1])
sig_g = garch_out.apply(lambda x: x[2])

garch_stats = pd.DataFrame({
    "df_garch": dfs_g,
    "mean_garch": mu_g,
    "std_garch": sig_g,
})

def es_factor(alpha, nu):
    q   = tdist.ppf(alpha, df=nu)
    pdf = tdist.pdf(q, df=nu)
    return -((nu + q**2) / (nu - 1)) * (pdf / alpha)

garch_stats["VaR95"] = (
    tdist.ppf(0.05, df=garch_stats["df_garch"]) * garch_stats["std_garch"]
    + garch_stats["mean_garch"]
)
garch_stats["ES95"] = (
    es_factor(0.05, garch_stats["df_garch"]) * garch_stats["std_garch"]
    + garch_stats["mean_garch"]
)

garch_sorted = garch_stats.sort_values("ES95")

worst_5_g = garch_sorted.head(5)
best_5_g  = garch_sorted.tail(5)

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Best 5 (lowest conditional tail risk)")
    st.dataframe(best_5_g[["VaR95", "ES95"]].applymap(lambda x: f"{x:.2%}"))

with c2:
    st.markdown("#### Worst 5 (highest conditional tail risk)")
    st.dataframe(worst_5_g[["VaR95", "ES95"]].applymap(lambda x: f"{x:.2%}"))

# ... rest of file unchanged ...
