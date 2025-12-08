# risk_stress_report.py

import pandas as pd

from utils.loaders import load_data
from risk.stress import (
    covid_crash_scenario,
    rate_shock_200bps_scenario,
    oil_spike_scenario,
    china_slowdown_scenario,
)


def scenario_pnl(df_ret: pd.DataFrame,
                 w: pd.Series,
                 scen_ret: pd.Series,
                 notional: float = 1_000_000) -> pd.Series:
    """
    Compute per-asset PnL under a given scenario.

    PnL_i = notional * w_i * r_i^{scenario}
    """
    scen_ret = scen_ret.reindex(df_ret.columns).fillna(0.0)
    w = w.reindex(df_ret.columns).fillna(0.0)

    pnl_by_asset = notional * w * scen_ret
    return pnl_by_asset


def print_scenario_result(name: str,
                          df_ret: pd.DataFrame,
                          w: pd.Series,
                          scen_ret: pd.Series,
                          notional: float = 1_000_000):
    """
    Pretty-print total PnL and top contributors for a scenario.
    """
    pnl_by_asset = scenario_pnl(df_ret, w, scen_ret, notional)
    total_pnl = pnl_by_asset.sum()

    print(f"\n=== {name} ===")
    print(f"Portfolio PnL ≈ {total_pnl:,.0f} €")

    # Worst 5 and best 5 assets in the scenario
    worst_5 = pnl_by_asset.sort_values().head(5)
    best_5 = pnl_by_asset.sort_values().tail(5)

    print("\n  Worst 5 assets (PnL):")
    for ticker, pnl in worst_5.items():
        print(f"    {ticker:<25} {pnl:>12,.0f} €")

    print("\n  Best 5 assets (PnL):")
    for ticker, pnl in best_5.items():
        print(f"    {ticker:<25} {pnl:>12,.0f} €")


def main():
    df_ret, w = load_data()

    scenarios = [
        ("COVID Crash",          covid_crash_scenario(df_ret)),
        ("+200bps Rate Shock",   rate_shock_200bps_scenario(df_ret)),
        ("Oil Spike",            oil_spike_scenario(df_ret)),
        ("China Slowdown",       china_slowdown_scenario(df_ret)),
    ]

    for name, scen_ret in scenarios:
        print_scenario_result(name, df_ret, w, scen_ret, notional=1_000_000)


if __name__ == "__main__":
    main()
