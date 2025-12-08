from utils.loaders import load_data
from risk.copulas import mc_portfolio_pnl, var_cvar
import numpy as np

df_ret, w = load_data()

pnl = mc_portfolio_pnl(df_ret, w, notional=1_000_000, n_sims=50_000, horizon_days=1)
print("Mean PnL:", np.mean(pnl))
print("Std PnL:", np.std(pnl))

var95, es95 = var_cvar(pnl, alpha=0.95)
var99, es99 = var_cvar(pnl, alpha=0.99)

print("VaR 95%:", var95 / 1_000_000)  # as %
print("ES 95% :", es95 / 1_000_000)
print("VaR 99%:", var99 / 1_000_000)
print("ES 99% :", es99 / 1_000_000)
