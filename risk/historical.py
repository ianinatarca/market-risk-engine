import numpy as np
import pandas as pd

def historical_var_es(returns, alpha=0.05):
    """
    Historical (non-parametric) VaR and ES for a return series.
    returns: 1D array-like or pandas Series of portfolio returns
    alpha: tail probability (e.g. 0.05 for 95% VaR)
    """
    # Convert to Series and drop NaNs
    r = pd.Series(returns).dropna()

    if len(r) == 0:
        return np.nan, np.nan

    # Historical VaR = empirical alpha-quantile
    var = r.quantile(alpha)

    # Historical ES = average of returns worse than VaR
    es = r[r <= var].mean()

    return var, es
