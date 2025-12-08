import numpy as np
import pandas as pd

def _align_weights_prices(w: pd.Series, prices: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Align and normalise weights and prices on the same index.
    """
    # ensure Series
    w = pd.Series(w).copy()
    prices = pd.Series(prices).copy()

    # align indices exactly
    w = w.reindex(prices.index)

    # fill missing weights with 0
    w = w.fillna(0.0)

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

    - Position €  = weight * notional
    - Equity delta (€/ +1%) for non-bond assets
    - DV01 (€/ +1bp) for bonds using a flat duration approximation

    Parameters
    ----------
    w : pd.Series
        Portfolio weights, indexed by asset name.
    prices : pd.Series
        Last price per asset, same index as returns.
    bond_mask : pd.Series[bool]
        True for bond-like instruments, False otherwise.
    notional : float
        Portfolio notional in EUR.
    duration_years : float
        Flat duration assumption for all bonds.

    Returns
    -------
    dict with:
        - "table": DataFrame with Position €, ΔPnL €/+1%, DV01 €/bp
        - "delta": per-asset equity delta
        - "dv01": per-asset DV01
        - "delta_total": sum of equity deltas
        - "dv01_total": sum of DV01
    """

    # 1) Align everything on the same index
    w_aligned, prices_aligned = _align_weights_prices(w, prices)
    idx = prices_aligned.index

    bond_mask = pd.Series(bond_mask, index=idx)
    bond_mask = bond_mask.fillna(False)

    # 2) Positions in EUR
    position_eur = w_aligned * notional

    # 3) Equity delta (€/ +1%): only for non-bonds
    eq_mask = ~bond_mask
    delta = position_eur * 0.01 * eq_mask  # linear approx: 1% of position

    # 4) DV01 for bonds (€/ +1bp)
    # Price change ≈ -Duration * Δy * Price
    # For 1bp (0.0001): ΔP ≈ -Duration * 0.0001 * Position
    dv01 = -position_eur * duration_years * 1e-4 * bond_mask

    # 5) Totals
    delta_total = delta.sum()
    dv01_total = dv01.sum()

    # 6) Per-asset table
    table = pd.DataFrame({
        "Position €": position_eur,
        "ΔPnL €/ +1.0%": delta,
        "DV01 €/bp": dv01,
    })

    return {
        "table": table,
        "delta": delta,
        "dv01": dv01,
        "delta_total": float(delta_total),
        "dv01_total": float(dv01_total),
    }


def apply_equity_shock(delta_total: float, shock_pct: float) -> float:
    """
    PnL from an equity shock, given total delta (€/ +1%).
    shock_pct is in %, e.g. +5 or -3.
    """
    return delta_total * (shock_pct / 1.0)


def apply_rate_shock(dv01_total: float, shock_bps: float) -> float:
    """
    PnL from a rate shock, given total DV01 (€/ +1bp).
    shock_bps is in basis points, e.g. +100 or -50.
    """
    return dv01_total * shock_bps
