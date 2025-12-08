# risk/copulas.py

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

from risk.ewma import compute_ewma_cov, cov_to_corr


def _to_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def mc_portfolio_pnl(
    df_ret: pd.DataFrame,
    weights: pd.Series,
    notional: float = 1_000_000,
    n_sims: int = 100_000,
    horizon_days: int = 1,
    lam: float = 0.94,
    nu_copula: int = 5,
    df_marg: int = 5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Monte Carlo simulation of portfolio PnL using a t-copula:

    Steps:
    1. Compute EWMA covariance of daily returns.
    2. Extract volatilities and correlation matrix.
    3. Simulate correlated t-copula shocks.
    4. Map shocks to Student-t margins with df_marg and scale so that
       marginal variance = 1.
    5. Scale by vol * sqrt(horizon_days) and add drift * horizon_days.
    6. Aggregate to portfolio and multiply by notional.

    Returns
    -------
    pnl : np.ndarray
        Simulated PnL in EUR, length = n_sims.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if isinstance(weights, pd.Series):
        w = weights.reindex(df_ret.columns).values
    else:
        w = np.asarray(weights)
    w = w / w.sum()  # ensure normalised

    R = df_ret.astype(float)

    # ---------- 1) EWMA covariance ----------
    cov_ewma = compute_ewma_cov(R, lam=lam)  # should return a covariance matrix (d x d)
    corr = cov_to_corr(cov_ewma)

    # Vols and dimension
    vols = np.sqrt(np.diag(cov_ewma))  # daily vols
    d = len(vols)

    # ---------- 2) Simulate t-copula ----------
    # Step 2.1: draw multivariate t with correlation 'corr' & df = nu_copula
    # Algorithm: Z = L * N / sqrt(Chi2(nu)/nu)
    L = np.linalg.cholesky(corr)

    # N ~ N(0, I)
    Z_norm = rng.standard_normal(size=(n_sims, d))
    # Chi2
    g = rng.chisquare(df=nu_copula, size=n_sims) / nu_copula
    g = np.sqrt(g).reshape(-1, 1)

    # Correlated t-copula factors
    Z = (Z_norm @ L.T) / g  # shape (n_sims, d)

    # Step 2.2: map to uniform via t CDF, then to Student-t(df_marg)
    U = tdist.cdf(Z, df=nu_copula)
    X = tdist.ppf(U, df=df_marg)

    # ---------- 3) Normalise margins to variance 1 ----------
    # Var of Student-t(df) is df / (df - 2) for df > 2
    var_t = df_marg / (df_marg - 2)
    X_std = X / np.sqrt(var_t)  # now each margin has approx var ~ 1

    # ---------- 4) Add drift & scale by vol * sqrt(horizon) ----------
    mu = R.mean().reindex(df_ret.columns).values  # daily mean returns

    # horizon mean + scaled shock
    # returns_sim shape: (n_sims, d)
    returns_sim = (
        mu * horizon_days
        + X_std * (vols * np.sqrt(horizon_days))
    )

    # ---------- 5) Portfolio aggregation ----------
    # portfolio return per scenario
    port_ret_sim = returns_sim @ w  # shape (n_sims,)
    pnl = notional * port_ret_sim   # in EUR

    return pnl


def var_cvar(pnl: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """
    Compute VaR and CVaR (expected shortfall) from a PnL distribution.

    Parameters
    ----------
    pnl : array-like
        Simulated PnL in EUR (negative = losses).
    alpha : float
        Confidence level (e.g. 0.99 → 99% VaR).

    Returns
    -------
    (VaR, CVaR) : tuple of floats
        Both returned as negative numbers (losses).
    """
    pnl = np.asarray(pnl)

    # left-tail probability e.g. 1% for 99% VaR
    p = 1.0 - alpha

    # VaR = quantile of PnL distribution at p (loss side)
    var_level = np.quantile(pnl, p)

    # CVaR = average PnL in tail ≤ VaR
    tail = pnl[pnl <= var_level]
    if tail.size == 0:
        es = var_level
    else:
        es = tail.mean()

    return var_level, es
