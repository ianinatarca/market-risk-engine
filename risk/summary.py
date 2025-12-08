# risk/summary.py

import numpy as np
import pandas as pd

from risk.student_t import compute_student_t_stats
from risk.portfolio_t import estimate_portfolio_df, portfolio_t_var_es
from risk.garch import garch_fit
from risk.portfolio_garch import portfolio_garch_var_es
from risk.historical import historical_var_es
from risk.copulas import mc_portfolio_pnl, var_cvar


def _detect_cols(stats_t: pd.DataFrame):
    """Helper to find ES/VaR/mean/std/df columns in compute_student_t_stats output."""
    def find_col(cols, must_contain):
        lc = [c.lower() for c in cols]
        for c, c_low in zip(cols, lc):
            if all(s.lower() in c_low for s in must_contain):
                return c
        raise KeyError(f"No column found containing {must_contain} in {cols}")

    cols = stats_t.columns.tolist()
    es95_col  = find_col(cols, ["es", "95"])
    var95_col = find_col(cols, ["var", "95"])
    mean_col  = find_col(cols, ["mean"])
    std_col   = find_col(cols, ["std"])
    dfs_col   = find_col(cols, ["df"])

    return {
        "es95": es95_col,
        "var95": var95_col,
        "mean": mean_col,
        "std": std_col,
        "df": dfs_col,
    }


def per_asset_static_es(df_ret: pd.DataFrame, alpha: float = 0.95, rng_seed: int = 42):
    """
    Fit per-asset Student-t and return worst/best 5 assets by ES.
    """
    rng = np.random.default_rng(rng_seed)
    stats_t = compute_student_t_stats(df_ret, rng)
    cols = _detect_cols(stats_t)
    es_col = cols["es95"]  # currently hard-coded to 95% ES

    stats_sorted = stats_t.sort_values(es_col)  # more negative = riskier
    worst_5 = stats_sorted.head(5)[
        [cols["mean"], cols["std"], cols["df"], cols["var95"], cols["es95"]]
    ]
    best_5 = stats_sorted.tail(5)[
        [cols["mean"], cols["std"], cols["df"], cols["var95"], cols["es95"]]
    ]

    return stats_t, worst_5, best_5


def portfolio_risk_summary(
    df_ret: pd.DataFrame,
    weights: pd.Series,
    notional: float = 1_000_000,
    rng_seed: int = 42,
):
    """
    Compute portfolio-level risk metrics with all four approaches:

    - Static Student-t (variance–covariance)
    - GARCH-t (conditional volatility, static correlation)
    - Historical (empirical)
    - Monte Carlo t-copula (non-Gaussian dependence)

    Returns a dictionary of results + best/worst assets from per-asset static t.
    """
    rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Portfolio return series
    # ------------------------------------------------------------------
    w = weights.reindex(df_ret.columns).fillna(0.0)
    w = w / w.sum()
    port_ret = df_ret @ w

    # ------------------------------------------------------------------
    # 1) Per-asset static t → best / worst 5
    # ------------------------------------------------------------------
    stats_t, worst_5_static, best_5_static = per_asset_static_es(df_ret, rng_seed=rng_seed)

    # ------------------------------------------------------------------
    # 2) Static portfolio Student-t
    # ------------------------------------------------------------------
    nu_p = estimate_portfolio_df(port_ret, rng)
    VaR95_t, ES95_t, VaR99_t, ES99_t = portfolio_t_var_es(port_ret, nu_p)

    # ------------------------------------------------------------------
    # 3) GARCH-t portfolio (per-asset GARCH, static corr)
    # ------------------------------------------------------------------
    garch_out = df_ret.apply(garch_fit)
    dfs_g = garch_out.apply(lambda x: x[0])
    mu_g  = garch_out.apply(lambda x: x[1])
    sig_g = garch_out.apply(lambda x: x[2])
    corr  = df_ret.corr().values

    VaR95_g, ES95_g, VaR99_g, ES99_g = portfolio_garch_var_es(
        w.values,
        mu_g.values,
        sig_g.values,
        corr,
        dfs_g.values,
    )

    # ------------------------------------------------------------------
    # 4) Historical VaR / ES (non-parametric, 1-day)
    # ------------------------------------------------------------------
    hist95, hist_es95 = historical_var_es(port_ret, alpha=0.05)  # 95% VaR
    hist99, hist_es99 = historical_var_es(port_ret, alpha=0.01)  # 99% VaR

    # ------------------------------------------------------------------
    # 5) Monte Carlo t-copula VaR / ES (1-day and 10-day)
    # ------------------------------------------------------------------
    pnl_1d = mc_portfolio_pnl(
        df_ret,
        w,
        notional=notional,
        n_sims=100_000,
        horizon_days=1,
        lam=0.94,
        nu_copula=5,
        df_marg=5,
    )
    var95_1d, es95_1d = var_cvar(pnl_1d, alpha=0.95)
    var99_1d, es99_1d = var_cvar(pnl_1d, alpha=0.99)

    pnl_10d = mc_portfolio_pnl(
        df_ret,
        w,
        notional=notional,
        n_sims=100_000,
        horizon_days=10,
        lam=0.94,
        nu_copula=5,
        df_marg=5,
    )
    var95_10d, es95_10d = var_cvar(pnl_10d, alpha=0.95)
    var99_10d, es99_10d = var_cvar(pnl_10d, alpha=0.99)

    return {
        "notional": notional,
        "port_ret": port_ret,
        "w": w,
        "static_t": {
            "nu_p": nu_p,
            "VaR95": VaR95_t,
            "ES95": ES95_t,
            "VaR99": VaR99_t,
            "ES99": ES99_t,
        },
        "garch_t": {
            "VaR95": VaR95_g,
            "ES95": ES95_g,
            "VaR99": VaR99_g,
            "ES99": ES99_g,
        },
        "historical": {
            "VaR95": hist95,
            "ES95": hist_es95,
            "VaR99": hist99,
            "ES99": hist_es99,
        },
        "mc_1d": {
            "VaR95": var95_1d,
            "ES95": es95_1d,
            "VaR99": var99_1d,
            "ES99": es99_1d,
        },
        "mc_10d": {
            "VaR95": var95_10d,
            "ES95": es95_10d,
            "VaR99": var99_10d,
            "ES99": es99_10d,
        },
        "per_asset_static": stats_t,
        "worst_5_static": worst_5_static,
        "best_5_static": best_5_static,
    }
