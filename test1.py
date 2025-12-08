from data.data_fetcher import load_price_history

df = load_price_history()
print(df.head())
print(df.tail())
print("Shape:", df.shape)
print("Date range:", df.index.min(), "→", df.index.max())
