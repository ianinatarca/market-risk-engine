import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.data_fetcher import load_price_history

# Load data
prices = load_price_history()
returns = np.log(prices / prices.shift(1)).dropna()

# Equal weights (modify later for real portfolio)
N = returns.shape[1]
w = np.ones(N) / N

# Portfolio returns
rp = returns.dot(w)

# Portfolio VaR (99%)
VaR_p = -np.percentile(rp, 1)

# Covariance
Sigma = np.cov(returns, rowvar=False)
sigma_p = np.sqrt(w.T @ Sigma @ w)

# Marginal VaR & Component VaR
mvar = (Sigma @ w) / sigma_p * VaR_p
cvar = w * mvar

assets = returns.columns

df = pd.DataFrame({
    "Component_VaR": cvar,
}, index=assets)

# Sort assets for plotting
df_sorted = df.sort_values("Component_VaR", ascending=False)

# Separate top & bottom contributors
top10 = df_sorted.head(10)
bottom10 = df_sorted.tail(10)

# ==== Plotting ====

plt.figure(figsize=(12, 6))
plt.bar(top10.index, top10["Component_VaR"], color="firebrick")
plt.title("Top 10 Portfolio VaR Contributors (Component VaR)")
plt.ylabel("Component VaR")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(bottom10.index, bottom10["Component_VaR"], color="seagreen")
plt.title("Bottom 10 Portfolio VaR Contributors (Diversifiers)")
plt.ylabel("Component VaR")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
