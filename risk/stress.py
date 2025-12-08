# risk/stress.py

import numpy as np
import pandas as pd


def _zero_scenario(df_ret: pd.DataFrame) -> pd.Series:
    """
    Helper: return a scenario with 0% shock for all assets.
    """
    return pd.Series(0.0, index=df_ret.columns)


def _is_bond(name: str) -> bool:
    """
    Very simple classifier for your data: treat instruments with coupons / %
    and typical bond tickers as 'bonds'.
    """
    name_low = name.lower()
    bond_keywords = [
        "%", "btp", "bund", "oat", "uk 0.125", "spain", "romania",
        "venture global lng",
    ]
    return any(k in name_low for k in bond_keywords)


def _is_commodity(name: str) -> bool:
    """
    Simple commodity classifier for your current portfolio.
    """
    name_low = name.lower()
    return any(k in name_low for k in ["gold", "uranium", "oil"])


def _is_china_sensitive(name: str) -> bool:
    """
    Very rough classification of China / EM / cyclical exposure.
    Extend as needed.
    """
    name_low = name.lower()
    keywords = [
        "miniso", "adaro", "shanghai", "china", "shenzhen",
        "shenghe", "xiaomi", "mercado libre", "nu holdings",
    ]
    return any(k in name_low for k in keywords)


def _is_defensive(name: str) -> bool:
    """
    Simple defensive / quality names.
    """
    name_low = name.lower()
    keywords = [
        "unitedhealth", "coca cola", "philip morris",
        "novo nordisk", "iqvia", "tata consumer",
    ]
    return any(k in name_low for k in keywords)


# --------------------------------------------------------------------
# 1. COVID-style equity crash
# --------------------------------------------------------------------

def covid_crash_scenario(df_ret: pd.DataFrame) -> pd.Series:
    """
    COVID-style cross-asset crash:

    - Global equities:  -25%
    - EM / cyclicals:   -35%
    - Commodities:      -20%
    - Gov bonds:        +5% (flight-to-quality)
    - Credit / others:  -10%

    Returns a Series of scenario returns indexed by df_ret.columns.
    """
    scen = _zero_scenario(df_ret)

    for name in df_ret.columns:
        if _is_bond(name):
            # Gov / IG bond rally
            scen[name] = 0.05
        elif _is_commodity(name):
            scen[name] = -0.20
        elif _is_china_sensitive(name):
            scen[name] = -0.35
        else:
            # Other risky assets (equities, credit, etc.)
            scen[name] = -0.25

    return scen


# --------------------------------------------------------------------
# 2. +200 bps parallel rate shock
# --------------------------------------------------------------------

def rate_shock_200bps_scenario(df_ret: pd.DataFrame) -> pd.Series:
    """
    +200 bps parallel shift in rates.

    Approximate price impact via simple 'effective duration' buckets:

    - Short/med gov bonds:   -8%
    - Long gov bonds:        -12%
    - Credit / others:       -5%
    - Non-rates assets:       0%

    Very stylised – adjust to your portfolio as needed.
    """
    scen = _zero_scenario(df_ret)

    for name in df_ret.columns:
        if not _is_bond(name):
            # We assume equities / commodities unchanged in this stylised test
            scen[name] = 0.0
        else:
            name_low = name.lower()
            # crude proxy: longer tenor → bigger loss
            if any(k in name_low for k in ["2035", "2036", "34", "38", "32"]):
                scen[name] = -0.12   # long end
            else:
                scen[name] = -0.08   # short/med

    return scen


# --------------------------------------------------------------------
# 3. Oil spike scenario
# --------------------------------------------------------------------

def oil_spike_scenario(df_ret: pd.DataFrame) -> pd.Series:
    """
    Oil spike / energy shock:

    - Energy / commodity names:        +20%
    - Cyclical equities & EM:          -15%
    - Defensives / staples / healthcare: -5%
    - Gov bonds:                       +2%

    Again, very stylised; classification is name-based.
    """
    scen = _zero_scenario(df_ret)

    for name in df_ret.columns:
        if _is_commodity(name) or "adaro" in name.lower():
            scen[name] = +0.20
        elif _is_bond(name):
            scen[name] = +0.02
        elif _is_china_sensitive(name):
            scen[name] = -0.15
        elif _is_defensive(name):
            scen[name] = -0.05
        else:
            scen[name] = -0.10

    return scen


# --------------------------------------------------------------------
# 4. China slowdown scenario
# --------------------------------------------------------------------

def china_slowdown_scenario(df_ret: pd.DataFrame) -> pd.Series:
    """
    China / EM growth slowdown:

    - China / EM / commodity cyclicals:   -20%
    - Global cyclicals / industrials:     -12%
    - Defensives / quality:               -3%
    - Core gov bonds:                     +3%
    """
    scen = _zero_scenario(df_ret)

    for name in df_ret.columns:
        name_low = name.lower()

        if _is_bond(name):
            scen[name] = +0.03
        elif _is_china_sensitive(name) or _is_commodity(name):
            scen[name] = -0.20
        elif any(k in name_low for k in ["siemens", "airbus", "caterpillar", "leonardo", "gerdau"]):
            scen[name] = -0.12
        elif _is_defensive(name):
            scen[name] = -0.03
        else:
            scen[name] = -0.08

    return scen

# --- ADD THIS AT THE BOTTOM OF risk/stress.py -----------------
import numpy as np
import pandas as pd

def _classify_asset(name: str) -> str:
    """
    Very simple asset classification based on name.
    Used to apply group shocks in stress scenarios.
    """
    n = name.lower()

    # Commodities / energy
    if "gold" in n or "uranium" in n or "adaro" in n:
        return "commodity"

    # Bonds (anything with a % coupon, BTP, OAT, Bund, UK gov, Spain, Romania, etc.)
    bond_keywords = ["%", "btp", "bund", "oat", "uk 0.125", "spain", "romania", "venture global lng"]
    if any(k in n for k in bond_keywords):
        return "bond"

    # Everything else = equity
    return "equity"


def _scenario_pnl(df_ret: pd.DataFrame,
                  weights: pd.Series,
                  group_shocks: dict,
                  notional: float = 1_000_000) -> pd.Series:
    """
    Apply simple percentage shocks by asset group and return per-asset PnL.

    PnL_i = notional * w_i * shock_group(asset_i)
    """
    # Align weights to columns
    w = weights.reindex(df_ret.columns).fillna(0.0)

    shocks = []
    for col in df_ret.columns:
        group = _classify_asset(col)
        shocks.append(group_shocks.get(group, 0.0))

    shocks = np.array(shocks)              # shape (n_assets,)
    pnl_values = notional * w.values * shocks
    return pd.Series(pnl_values, index=df_ret.columns, name="PnL")


def run_all_stress(df_ret: pd.DataFrame,
                   weights: pd.Series,
                   notional: float = 1_000_000) -> dict:
    """
    Run all stress scenarios used by the dashboard.

    Returns
    -------
    dict:
        {
          scenario_name: {
              "portfolio_pnl": float,
              "asset_pnl": pd.Series,
              "worst5": pd.Series,
              "best5": pd.Series,
          },
          ...
        }
    """

    # You can tweak these shocks later if you want to calibrate them
    scenarios = {
        "COVID Crash": {
            "equity": -0.30,
            "commodity": -0.25,
            "bond": +0.05,
        },
        "+200bps Rate Shock": {
            "equity": -0.05,
            "commodity": -0.05,
            "bond": -0.08,
        },
        "Oil Spike": {
            "equity": -0.10,
            "commodity": +0.20,   # gold / uranium / Adaro win
            "bond": -0.03,
        },
        "China Slowdown": {
            "equity": -0.15,
            "commodity": -0.10,
            "bond": +0.03,
        },
    }

    results = {}
    for name, group_shocks in scenarios.items():
        pnl_series = _scenario_pnl(df_ret, weights, group_shocks, notional=notional)
        results[name] = {
            "portfolio_pnl": float(pnl_series.sum()),
            "asset_pnl": pnl_series,
            "worst5": pnl_series.nsmallest(5),
            "best5": pnl_series.nlargest(5),
        }

    return results
# --------------------------------------------------------------
