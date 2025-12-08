import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "ims_clean.csv"

def load_price_history() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        decimal=",",
        thousands=".",
    )

    # Parse Date (now like 21112025 = 21-11-2025)
    df["Date"] = pd.to_datetime(df["Date"], format="%d%m%Y")

    # Set index & sort
    df = df.set_index("Date").sort_index()

    # Convert prices to floats
    df = df.apply(pd.to_numeric, errors="coerce")

    # Forward-fill
    df = df.ffill()

    # Last 6 months only
    six_months_ago = df.index.max() - pd.DateOffset(months=6)
    df = df[df.index >= six_months_ago]

    return df
