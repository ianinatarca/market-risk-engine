import pandas as pd
import numpy as np

def load_data():
    # 1) Load PRICES with correct separators & numeric parsing
    df_prices = pd.read_csv(
        "data/ims_clean.csv",
        sep=";",             # semicolon-separated
        index_col=0,         # first column = Date
        decimal=",",         # 46,91 -> 46.91
        thousands=".",       # 3.531,7435 -> 3531.7435
    )

    # Your dates look like "21112025" → use "%d%m%Y"
    df_prices.index = pd.to_datetime(df_prices.index.astype(str), format="%d%m%Y")

    # 2) Force all columns to numeric (in case anything slipped as string)
    df_prices = df_prices.apply(pd.to_numeric, errors="coerce")

    # 3) Sort by date (just in case)
    df_prices = df_prices.sort_index()

    # 4) Convert prices → log returns
    df_ret = np.log(df_prices / df_prices.shift(1)).dropna(how="all")

    # 5) Load portfolio weights
    alloc = pd.read_csv("data/weights.csv")

    # Convert to Series indexed by ticker
    w = alloc.set_index("Ticker")["Weight"]

    # Align weights to df_ret columns; missing tickers → 0
    w = w.reindex(df_ret.columns).fillna(0.0)

    # Normalize to sum to 1
    if w.sum() == 0:
        raise ValueError("All weights are zero after alignment – check names in weights.csv vs ims_clean.csv")
    w = w / w.sum()

    return df_ret, w
