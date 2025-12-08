from pathlib import Path
import pandas as pd
from data.data_fetcher import DATA_PATH, load_price_history

print("DATA_PATH:", DATA_PATH)
print("Exists on disk?:", Path(DATA_PATH).exists())

print("\n--- Raw first 5 lines of file ---")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip())

print("\n--- Trying to load with pandas.read_csv ---")
try:
    df_raw = pd.read_csv(
        DATA_PATH,
        sep=";",
        decimal=",",
        thousands=".",
    )
    print("Raw df shape:", df_raw.shape)
    print("Raw columns:", list(df_raw.columns))
    print("\nFirst 3 rows:")
    print(df_raw.head(3))
except Exception as e:
    print("\nError in pd.read_csv:", repr(e))

print("\n--- Trying load_price_history() ---")
try:
    df = load_price_history()
    print("Processed df shape:", df.shape)
    print("Index:", df.index.min(), "→", df.index.max())
    print("Columns:", list(df.columns))
except Exception as e:
    print("\nError in load_price_history():", repr(e))
