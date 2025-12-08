# risk/sensitivities.py

import pandas as pd


def compute_position_values(weights: pd.Series,
                            prices: pd.Series,
                            notional: float = 1_000_000) -> pd.Series:
    """
    Map portfolio weights to €-exposures.

    We ignore 'prices' for now and just use weights * notional.
    """
    w = weights.reindex(prices.index).fillna(0.0)
    pos = w * notional
    pos.name = "Position €"
    return pos


def compute_equity_delta(position_values: pd.Series,
                         bond_mask: pd.Series,
                         pct_move: float = 1.0) -> pd.Series:
    """
    Simple equity delta:
        ΔPnL_i = Position_i * (pct_move / 100)

    Bonds are excluded (delta = 0) because they are handled via DV01.
    """
    delta = position_values.copy().astype(float)
    delta[bond_mask] = 0.0
    delta *= pct_move / 100.0
    delta.name = f"ΔPnL €/ +{pct_move}%"
    return delta


def compute_dv01(position_values: pd.Series,
                 bond_mask: pd.Series,
                 duration_years: float = 6.0) -> pd.Series:
    """
    Approximate DV01 using a flat duration for all bonds:

        DV01_i ≈ - Position_i * Duration / 10_000

    (negative: higher yields → lower prices)
    """
    dv01 = pd.Series(0.0, index=position_values.index, name="DV01 €/bp")
    dv01[bond_mask] = -position_values[bond_mask] * duration_years / 10_000.0
    return dv01


def total_sensitivities(weights: pd.Series,
                        prices: pd.Series,
                        bond_mask: pd.Series,
                        notional: float = 1_000_000,
                        duration_years: float = 6.0) -> dict:
    """
    Main wrapper used by the dashboard.

    Returns:
      - table      : per-asset Position / Delta / DV01
      - delta      : per-asset equity delta
      - dv01       : per-asset DV01
      - delta_total: portfolio equity delta (€/1%)
      - dv01_total : portfolio DV01 (€/bp)
    """
    pos = compute_position_values(weights, prices, notional)
    delta = compute_equity_delta(pos, bond_mask, pct_move=1.0)
    dv01 = compute_dv01(pos, bond_mask, duration_years)

    table = pd.concat([pos, delta, dv01], axis=1)

    return {
        "table": table,
        "delta": delta,
        "dv01": dv01,
        "delta_total": delta.sum(),
        "dv01_total": dv01.sum(),
    }


def apply_equity_shock(delta_total: float, pct_move: float) -> float:
    """
    Portfolio PnL for a given equity move (in %):

        PnL ≈ delta_total * pct_move
    """
    return delta_total * pct_move


def apply_rate_shock(dv01_total: float, bp_move: float) -> float:
    """
    Portfolio PnL for a parallel rate shift (in basis points):

        PnL ≈ dv01_total * bp_move
    """
    return dv01_total * bp_move
