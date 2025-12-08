# risk/sensitivities.py

import numpy as np
import pandas as pd


def _align_weights_prices(w: pd.Series, prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Align and normalise weights and prices on the same index.
    Assumes both indices are already 'clean' (e.g. lower-case, stripped).
    """
    w = pd.Series(w).copy()
    prices = pd.Series(prices).copy()

    # align on the union of names, fill missing weights with 0
    w = w.reindex(prices.index).fillna(0.0)

    return w, prices


def total_sensitivities(
    w: pd.Series,
    prices: pd.Series,
    bond_mask: pd.Series,
    notional: float = 1_000_000.0,
    duration_years: float = 6.0,
):
    """
    Compute per-asset and total sensitivities:

    - Position €        = weight * notional
    - Equity delta      = ΔPnL €/+1% for non-bonds
    - DV01 €/bp         = duration-based for bonds

    Parameters
    ----------
    w : pd.Series
        Portfolio weights (already aligned + normalised names).
    prices : pd.Series
        Current prices, same index as returns.
    bond_mask : pd.Series[bool]
        True for bond-like instruments.
    notional : float
        Portfolio notional in EUR.
    duration_years : float
        Flat duration assumption for all bonds.

    Returns
    -------
    dict with:
        - "table": DataFrame with Position €, ΔPnL €/+1.0%, DV01 €/bp
        - "delta": per-asset equity delta
        - "dv01": per-asset DV01
        - "delta_total": sum of equity deltas
        - "dv01_total": sum of DV01
    """

    # 1) Align on the same index
    w_aligned, prices_aligned = _align_weights_prices(w, prices)
    idx = prices_aligned.index

    # make sure bond_mask has same index
    bond_mask = pd.Series(bond_mask, index=idx).fillna(False)

    # 2) Positions in EUR
    position_eur = w_aligned * notional

    # 3) Equity delta (€/ +1%) – only for non-bonds
    eq
