import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.loaders import load_data

# ============================================================
# 1. Load data (returns + weights)
# ============================================================

df_ret, w = load_data()      # df_ret = log returns (T × N), w = weights (N)

# ============================================================
# 2. Compute portfolio return & PnL
# ============================================================

# Portfolio return: r_t = Σ_i w_i * r_{t,i}
port_ret = df_ret.mul(w, axis=1).sum(axis=1)
port_ret.name = "Portfolio Return"

# PnL for notional = 1,000,000 €
notional = 1_000_000
pnl = port_ret * notional
pnl.name = "PnL (EUR)"

# ============================================================
# 3. Cumulative return
# ============================================================

# Cumulative performance curve
cum_ret = np.exp(port_ret.cumsum()) - 1
cum_ret.name = "Cumulative Return"

# Final cumulative performance
final_cum = cum_ret.iloc[-1]

# ============================================================
# 4. Daily & annualized statistics
# ============================================================

# Daily mean return
mean_daily = port_ret.mean()

# Daily volatility
vol_daily = port_ret.std()

# Annualization factor (≈ 252 trading days)
annual_factor = 252

mean_annual = mean_daily * annual_factor
vol_annual = vol_daily * np.sqrt(annual_factor)

# ============================================================
# 5. Sharpe Ratio (risk-free assumed 0)
# ============================================================

sharpe = mean_annual / vol_annual

# ============================================================
# 6. Sortino Ratio
# ============================================================

# downside deviation = volatility of negative returns
downside_dev = port_ret[port_ret < 0].std() * np.sqrt(annual_factor)
sortino = mean_annual / downside_dev if downside_dev > 0 else np.nan

# ============================================================
# 7. Summary Table
# ============================================================

summary = pd.DataFrame({
    "Daily Mean Return": [mean_daily],
    "Daily Volatility": [vol_daily],
    "Annual Mean Return": [mean_annual],
    "Annual Volatility": [vol_annual],
    "Final Cumulative Return": [final_cum],
    "Sharpe Ratio": [sharpe],
    "Sortino Ratio": [sortino],
})

print("\n=== PORTFOLIO PERFORMANCE SUMMARY ===")
print(summary.round(6))

# ============================================================
# 8. Optional: Plot performance
# ============================================================

plt.figure(figsize=(10,5))
plt.plot(cum_ret, label="Cumulative Return")
plt.title("Portfolio Cumulative Return")
plt.xlabel("Date")
plt.ylabel("Return")
plt.grid(True)
plt.legend()
plt.show()
