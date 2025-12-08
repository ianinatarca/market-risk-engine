import numpy as np
import pandas as pd
from data.data_fetcher import load_price_history

# Load data
prices = load_price_history()
returns = np.log(prices / prices.shift(1)).dropna()

# Equal weights (or load portfolio weights later)
N = returns.shape[1]
w = np.ones(N) / N

# Portfolio returns
rp = returns.dot(w)

# Portfolio VaR 99%
VaR_p = -np.percentile(rp, 1)

# Covariance matrix
Sigma = np.cov(returns, rowvar=False)

# Portfolio volatility
sigma_p = np.sqrt(w.T @ Sigma @ w)

# Compute Marginal VaR
mvar = (Sigma @ w) / sigma_p * VaR_p

# Component VaR
cvar = w * mvar

# Build summary table
assets = returns.columns
df = pd.DataFrame({
    "Weight": w,
    "Marginal_VaR": mvar,
    "Component_VaR": cvar,
    "%_VaR_Contribution": cvar / VaR_p * 100
}, index=assets)

# Sort by Component VaR
df_sorted = df.sort_values("Component_VaR", ascending=False)

print("\n=== TOP RISK CONTRIBUTORS ===")
print(df_sorted.head(10))

print("\n=== LOWEST RISK CONTRIBUTORS (DIVERSIFIERS) ===")
print(df_sorted.tail(10))
