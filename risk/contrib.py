import numpy as np
import pandas as pd

def component_es(returns_matrix, weights, alpha=0.05):
    """
    Compute Marginal ES and Component ES via simulation.
    
    Parameters:
        returns_matrix: np.array, shape (N_sims, N_assets)
        weights: np.array, shape (N_assets,)
        alpha: tail probability (0.05 for ES95)
    
    Returns:
        DataFrame with MES, CES, pct_contrib
    """

    # Portfolio returns
    pnl = returns_matrix @ weights

    # ES threshold: worst alpha% scenarios
    cutoff = np.percentile(pnl, 100 * alpha)

    # Mask tail events
    tail_idx = pnl <= cutoff
    tail_returns = returns_matrix[tail_idx]
    tail_pnl = pnl[tail_idx]

    # Portfolio ES
    ES = tail_pnl.mean()

    # Covariance inside the tail
    cov_tail = np.cov(tail_returns.T, tail_pnl, ddof=0)

    # Last column is cov(asset_i, pnl)
    cov_with_port = cov_tail[:-1, -1]

    # Marginal ES
    MES = cov_with_port / (alpha * ES)

    # Component ES
    CES = weights * MES

    df = pd.DataFrame({
        "weight": weights,
        "MES": MES,
        "CES": CES,
        "pct_contrib": CES / ES
    })

    return df
