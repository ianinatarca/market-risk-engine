# risk/returns.py

import numpy as np
import pandas as pd

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()
