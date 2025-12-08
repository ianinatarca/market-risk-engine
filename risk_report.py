import numpy as np
import pandas as pd
from scipy.stats import t as tdist

from utils.loaders import load_data
from risk.ewma import compute_ewma_cov, cov_to_corr
from risk.student_t import compute_student_t_stats
from risk.portfolio_t import estimate_portfolio_df, portfolio_t_var_es
from risk.garch import garch_fit
from risk.portfolio_garch import portfolio_garch_var_es
from risk.historical import historical_var_es


# ============================================================
# 0. Setup & data
# ============================================================

rng = np.random.default_rng(42)

# df = returns (log returns), weights = portfolio weights aligned to df.columns
df, weights = load_data()


# New: EWMA covariance + correlation
cov_ewma = compute_ewma_cov(df, lam=0.94)
corr_ewma = cov_to_corr(cov_ewma)


# Portfolio return series
port_ret = df @ weights


# ============================================================
# 1. Static per-asset Student-t stats & ES ranking
# ============================================================

print("Computing static Student-t per-asset risk...")
stats_t = compute_student_t_stats(df, rng)

print("\n=== Per-asset static Student-t ES (95%) ===")
print("Columns in stats_t:", stats_t.columns.tolist())

# ---- robustly detect column names ----
def find_col(cols, must_contain):
    """
    cols: list of column names
    must_contain: list of substrings that must all appear (case-insensitive)
    returns: first matching column name, or raises if none
    """
    lc = [c.lower() for c in cols]
    for c, c_low in zip(cols, lc):
        if all(s.lower() in c_low for s in must_contain):
            return c
    raise KeyError(f"No column found containing {must_contain} in {cols}")

# detect key columns
es95_col   = find_col(stats_t.columns, ["es", "95"])
var95_col  = find_col(stats_t.columns, ["var", "95"])
mean_col   = find_col(stats_t.columns, ["mean"])
std_col    = find_col(stats_t.columns, ["std"])
dfs_col    = find_col(stats_t.columns, ["df"])

print("Detected columns →",
      f"mean={mean_col}, std={std_col}, dfs={dfs_col}, VaR95={var95_col}, ES95={es95_col}")

# Sort by ES 95% (more negative = riskier)
stats_t_sorted = stats_t.sort_values(es95_col)

worst_5_t = stats_t_sorted.head(5)
best_5_t  = stats_t_sorted.tail(5)

print("\n--- Worst 5 assets by ES 95% (static Student-t) ---")
print(
    worst_5_t[[mean_col, std_col, dfs_col, var95_col, es95_col]]
)

print("\n--- Best 5 assets by ES 95% (static Student-t) ---")
print(
    best_5_t[[mean_col, std_col, dfs_col, var95_col, es95_col]]
)


# ============================================================
# 2. Portfolio Student-t VaR / ES
# ============================================================

print("\nEstimating portfolio df via KS (Student-t)...")
nu_p = estimate_portfolio_df(port_ret, rng)

print("Computing portfolio Student-t VaR/ES...")
VaR95_t, ES95_t, VaR99_t, ES99_t = portfolio_t_var_es(port_ret, nu_p)


# ============================================================
# 3. Per-asset GARCH-t, portfolio GARCH VaR / ES
# ============================================================

print("\nEstimating GARCH models per asset...")
garch_out = df.apply(garch_fit)
# stats_t is from your static t fit



# garch_out is assumed to be a Series of [nu_garch, mean_g, std_g]
dfs_g = garch_out.apply(lambda x: x[0])
mu_g  = garch_out.apply(lambda x: x[1])
sig_g = garch_out.apply(lambda x: x[2])

print("Computing portfolio GARCH VaR/ES...")
#corr = df.corr().values
corr = corr_ewma.values


VaR95_g, ES95_g, VaR99_g, ES99_g = portfolio_garch_var_es(
    weights.values,
    mu_g.values,
    sig_g.values,
    corr,
    dfs_g.values,
)


# ============================================================
# 4. Historical (non-parametric) VaR / ES
# ============================================================

hist95, hist_es95 = historical_var_es(port_ret, 0.05)
hist99, hist_es99 = historical_var_es(port_ret, 0.01)


# ============================================================
# 5. Print portfolio-level summary
# ============================================================

print("\n--- PORTFOLIO RESULTS ---")

print("\nStatic Student-t (unconditional):")
print(f"VaR95 = {VaR95_t:.2%}, ES95 = {ES95_t:.2%}")
print(f"VaR99 = {VaR99_t:.2%}, ES99 = {ES99_t:.2%}")

print("\nGARCH-t (conditional):")
print(f"VaR95 = {VaR95_g:.2%}, ES95 = {ES95_g:.2%}")
print(f"VaR99 = {VaR99_g:.2%}, ES99 = {ES99_g:.2%}")

print("\nHistorical (non-parametric):")
print(f"VaR95 = {hist95:.2%}, ES95 = {hist_es95:.2%}")
print(f"VaR99 = {hist99:.2%}, ES99 = {hist_es99:.2%}")

# Optionally scale to 1M notional
V0 = 1_000_000
print("\nScaled to €1,000,000 notional (1-day losses):")
print(f"Static t  : VaR95 = {VaR95_t * V0:,.0f}€, ES95 = {ES95_t * V0:,.0f}€")
print(f"GARCH-t   : VaR95 = {VaR95_g * V0:,.0f}€, ES95 = {ES95_g * V0:,.0f}€")
print(f"Historical: VaR95 = {hist95   * V0:,.0f}€, ES95 = {hist_es95   * V0:,.0f}€")


# ============================================================
# 6. Per-asset GARCH-t ES95 ranking
# ============================================================

print("\n=== Per-asset conditional GARCH-t ES (95%) ===")

# Build per-asset GARCH stats table
garch_stats = pd.DataFrame({
    "df_garch":   dfs_g,
    "mean_garch": mu_g,
    "std_garch":  sig_g,
})

def es_factor(alpha, nu):
    """
    Returns the ES multiplier for a Student-t(ν) at tail prob alpha.
    nu can be a scalar or a pandas Series.
    """
    q = tdist.ppf(alpha, df=nu)
    pdf = tdist.pdf(q, df=nu)
    return -((nu + q**2) / (nu - 1)) * (pdf / alpha)

# 1-day conditional VaR / ES at 95%
garch_stats["VaR 95"] = (
    tdist.ppf(0.05, df=garch_stats["df_garch"]) * garch_stats["std_garch"]
    + garch_stats["mean_garch"]
)
garch_stats["ES 95"] = (
    es_factor(0.05, garch_stats["df_garch"]) * garch_stats["std_garch"]
    + garch_stats["mean_garch"]
)

# Sort by ES 95 (most negative = riskiest)
garch_sorted = garch_stats.sort_values("ES 95")

worst_5_g = garch_sorted.head(5)
best_5_g  = garch_sorted.tail(5)

print("\n--- Worst 5 assets by ES 95% (GARCH-t) ---")
print(worst_5_g[["df_garch", "mean_garch", "std_garch", "VaR 95", "ES 95"]])

print("\n--- Best 5 assets by ES 95% (GARCH-t) ---")
print(best_5_g[["df_garch", "mean_garch", "std_garch", "VaR 95", "ES 95"]])
