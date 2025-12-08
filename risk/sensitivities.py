# risk/sensitivities.py

import numpy as np
import pandas as pd


def _align_weights_prices(w: pd.Series, prices: pd.Series):
    """
    Align weights and prices on the same index.
    Missing weights become 0.
    """
    w = pd.Series(w).copy()
    prices = pd.Series(prices).copy()

    # Align
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
    Compute equity deltas and DV01 for bonds.
    """

    # Ensure aligned & consistent indices
    w, prices = _align_weights_prices(w, prices)
    idx = prices.index

    # Ensure mask is a boolean Series on correct index
    bond_mask = pd.Series(bond_mask, index=idx).fillna(False)

    # Position € for each asset
    position_eur = w * notional

    # Equity delta (€/ +1%)
    eq_mask = ~bond_mask
    delta = position_eur * 0.01 * eq_mask  # +1% move

    # DV01 for bonds (€/ +1bp)
    dv01 = -position_eur * duration_years * 1e-4 * bond_mask  # 1bp = 0.0001

    # Totals
    delta_total = float(delta.sum())
    dv01_total = float(dv01.sum())

    # Table
    table = pd.DataFrame(
        {
            "Position €": position_eur,
            "ΔPnL €/ +1.0%": delta,
            "DV01 €/bp": dv01,
        },
        index=idx,
    )

    return {
        "table": table,
        "delta": delta,
        "dv01": dv01,
        "delta_total": delta_total,
        "dv01_total": dv01_total,
    }


def apply_equity_shock(delta_total: float, shock_pct: float):
    """Return PnL from equity shock."""
    return delta_total * (shock_pct / 1.0)


def apply_rate_shock(dv01_total: float, shock_bps: float):
    """Return PnL from rate shock."""
    return dv01_total * shock_bps
