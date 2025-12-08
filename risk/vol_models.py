# risk/vol_models.py

import numpy as np
import pandas as pd

# Simple EWMA volatility, like RiskMetrics
def ewma_vol(series: pd.Series, lam: float = 0.94) -> float:
    """Compute EWMA volatility for one return series."""
    r = series.dropna().values
    if len(r) == 0:
        return np.nan
    weights = np.array([(1 - lam) * lam**i for i in range(len(r))])[::-1]
    var = np.sum(weights * r**2)
    return float(np.sqrt(var))

def vol_vector(returns: pd.DataFrame, method: str = "ewma") -> pd.Series:
    """
    Compute a volatility estimate for each asset in the returns DataFrame.
    Currently only EWMA is implemented.
    """
    vols = {}
    for col in returns.columns:
        if method == "ewma":
            vols[col] = ewma_vol(returns[col])
        else:
            raise ValueError("Unknown vol method: " + method)
    return pd.Series(vols)
