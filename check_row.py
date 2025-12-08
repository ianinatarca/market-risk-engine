from data.data_fetcher import load_price_history
import numpy as np

df = load_price_history()
print(df.head(10))
print(df.tail(10))

returns = np.log(df / df.shift(1))
print("\nReturns head:")
print(returns.head())
print("\nReturns shape:", returns.shape)
