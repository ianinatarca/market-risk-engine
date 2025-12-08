# risk/ewma.py

import numpy as np
import pandas as pd

def compute_ewma_cov(df_ret: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """
    Compute EWMA covariance matrix for a panel of returns.

    Parameters
    ----------
    df_ret : DataFrame
        T x N matrix of returns (rows = dates, cols = tickers).
    lam : float
        Decay factor lambda in (0,1). Larger = slower decay (longer memory).

    Returns
    -------
    cov_ewma : DataFrame
        N x N EWMA covariance matrix at the last date.
    """
    R = df_ret.values  # shape (T, N)
    T, N = R.shape

    if T < 2:
        raise ValueError("Need at least 2 observations to compute EWMA covariance.")

    # Initialize with sample covariance of first ~30 days (or all if less)
    init_window = min(30, T)
    S = np.cov(R[:init_window].T, ddof=1)  # N x N

    # Iterate EWMA recursion from init_window to end
    for t in range(init_window, T):
        r_t = R[t, :].reshape(-1, 1)          # N x 1
        S = lam * S + (1.0 - lam) * (r_t @ r_t.T)

    # Wrap as DataFrame
    cov_ewma = pd.DataFrame(S, index=df_ret.columns, columns=df_ret.columns)
    return cov_ewma


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """
    Convert covariance matrix to correlation matrix.
    """
    std = np.sqrt(np.diag(cov.values))
    # Avoid division by zero
    std[std == 0] = 1e-12
    D_inv = np.diag(1.0 / std)
    corr = D_inv @ cov.values @ D_inv
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)
