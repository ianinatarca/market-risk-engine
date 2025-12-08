# risk/backtest.py

import numpy as np
import pandas as pd
from scipy.stats import chi2


# ------------------------------------------------------------
# 0. Helpers
# ------------------------------------------------------------

def _to_1d_array(x):
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


# ------------------------------------------------------------
# 1. Rolling Historical VaR
# ------------------------------------------------------------

def rolling_historical_var(returns, alpha=0.99, window=250):
    """
    Rolling historical VaR (left-tail quantile of past 'window' returns).

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns.
    alpha : float
        VaR confidence level (e.g. 0.99 for 99% VaR).
    window : int
        Lookback window size.

    Returns
    -------
    pd.Series of VaR values (same index as returns), with NaN for the
    first `window` observations.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    p = 1.0 - alpha  # left-tail probability, e.g. 0.01 for 99% VaR

    var_values = []
    idx = returns.index

    for i in range(len(returns)):
        if i < window:
            var_values.append(np.nan)
        else:
            window_slice = returns.iloc[i - window : i]
            # Left-tail quantile (loss, typically negative)
            q = np.quantile(window_slice.values, p)
            var_values.append(q)

    return pd.Series(var_values, index=idx, name=f"VaR_hist_{int(alpha*100)}")


# ------------------------------------------------------------
# 2. Kupiec POF test
# ------------------------------------------------------------

def _kupiec_pof(breaches, alpha):
    """
    Kupiec Proportion of Failures (POF) test.

    H0: the observed breach probability equals 1 - alpha.
    """
    b = _to_1d_array(breaches).astype(bool)
    n = b.size
    x = int(b.sum())
    if n == 0:
        return {"n": 0, "x": 0, "pi_hat": np.nan, "LR_pof": np.nan, "p_value": np.nan}

    pi_hat = x / n
    p0 = 1.0 - alpha  # expected breach prob

    eps = 1e-10
    pi_hat_c = np.clip(pi_hat, eps, 1.0 - eps)

    logL0 = x * np.log(p0) + (n - x) * np.log(1.0 - p0)
    logL1 = x * np.log(pi_hat_c) + (n - x) * np.log(1.0 - pi_hat_c)

    LR = -2.0 * (logL0 - logL1)
    pval = 1.0 - chi2.cdf(LR, df=1)

    return {
        "n": n,
        "x": x,
        "pi_hat": pi_hat,
        "LR_pof": LR,
        "p_value": pval,
    }


# ------------------------------------------------------------
# 3. Christoffersen Independence test
# ------------------------------------------------------------

def _christoffersen_ind(breaches):
    """
    Christoffersen test for independence of exceptions.

    H0: breaches are independent over time (no clustering).
    """
    b = _to_1d_array(breaches).astype(bool)
    if b.size < 2:
        return {
            "n00": 0, "n01": 0, "n10": 0, "n11": 0,
            "pi01": np.nan, "pi11": np.nan,
            "LR_ind": np.nan, "p_value": np.nan,
        }

    b_prev = b[:-1]
    b_curr = b[1:]

    n00 = np.sum((b_prev == 0) & (b_curr == 0))
    n01 = np.sum((b_prev == 0) & (b_curr == 1))
    n10 = np.sum((b_prev == 1) & (b_curr == 0))
    n11 = np.sum((b_prev == 1) & (b_curr == 1))

    if (n00 + n01 == 0) or (n10 + n11 == 0):
        return {
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "pi01": np.nan, "pi11": np.nan,
            "LR_ind": np.nan, "p_value": np.nan,
        }

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)

    n0 = n00 + n01
    n1 = n10 + n11
    pi = (n01 + n11) / (n0 + n1)

    eps = 1e-10
    pi01_c = np.clip(pi01, eps, 1.0 - eps)
    pi11_c = np.clip(pi11, eps, 1.0 - eps)
    pi_c = np.clip(pi, eps, 1.0 - eps)

    logL_r = n0 * np.log(1.0 - pi_c) + n1 * np.log(pi_c)
    logL_u = (
        n00 * np.log(1.0 - pi01_c) +
        n01 * np.log(pi01_c) +
        n10 * np.log(1.0 - pi11_c) +
        n11 * np.log(pi11_c)
    )

    LR = -2.0 * (logL_r - logL_u)
    pval = 1.0 - chi2.cdf(LR, df=1)

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi01": pi01, "pi11": pi11,
        "LR_ind": LR,
        "p_value": pval,
    }


# ------------------------------------------------------------
# 4. Christoffersen Conditional Coverage
# ------------------------------------------------------------

def _christoffersen_cc(breaches, alpha):
    """
    Christoffersen Conditional Coverage test.

    Combines:
    - Kupiec POF (correct unconditional frequency)
    - Independence (no clustering)

    H0: correct frequency AND independence.
    """
    pof = _kupiec_pof(breaches, alpha)
    ind = _christoffersen_ind(breaches)

    LR_pof = pof["LR_pof"]
    LR_ind = ind["LR_ind"]

    if np.isnan(LR_ind):
        LR_cc = np.nan
        pval = np.nan
    else:
        LR_cc = LR_pof + LR_ind
        pval = 1.0 - chi2.cdf(LR_cc, df=2)

    return {
        "pof": pof,
        "ind": ind,
        "LR_cc": LR_cc,
        "p_value": pval,
    }


# ------------------------------------------------------------
# 5. Basel traffic light helper (99% VaR)
# ------------------------------------------------------------

def basel_thresholds_99(n):
    """
    Approximate Basel traffic-light thresholds for 99% VaR.

    Official (for n = 250):
        GREEN : 0–4
        YELLOW: 5–9
        RED   : >=10

    For other n, we scale thresholds approximately as:
        green_max  ~ 4 * n / 250
        yellow_max ~ 9 * n / 250

    and round to the nearest integer.
    """
    if n <= 0:
        return (None, None)

    # For very small samples, classification is not very meaningful.
    if n < 80:
        return (None, None)

    green_max = int(np.round(4.0 * n / 250.0))
    yellow_max = int(np.round(9.0 * n / 250.0))

    # Ensure monotonicity
    green_max = max(green_max, 0)
    yellow_max = max(yellow_max, green_max)

    return (green_max, yellow_max)


def basel_traffic_light_99(n, x):
    """
    Basel traffic light regime for 99% VaR.

    Returns
    -------
    "GREEN", "YELLOW", "RED", or "UNDEFINED" if sample too small.
    """
    if n <= 0:
        return "UNDEFINED"

    green_max, yellow_max = basel_thresholds_99(n)
    if green_max is None:
        # Not enough data to meaningfully classify
        return "UNDEFINED"

    if x <= green_max:
        return "GREEN"
    elif x <= yellow_max:
        return "YELLOW"
    else:
        return "RED"


# ------------------------------------------------------------
# 6. Main backtest wrapper
# ------------------------------------------------------------

def backtest_var(returns, var_series, alpha=0.99):
    """
    Backtest a VaR series against realized portfolio returns.

    Parameters
    ----------
    returns : pd.Series
        Realized portfolio returns.
    var_series : pd.Series
        VaR estimates (same index). Should be negative numbers (loss).
    alpha : float
        VaR confidence level, e.g. 0.99.

    Returns
    -------
    dict with:
        t_obs           : number of observations used
        n_exceptions    : number of breaches
        exception_rate  : n_exceptions / t_obs
        breaches        : pd.Series[bool]
        zone            : Basel traffic-light (99% only)
        thresholds      : (green_max, yellow_max) or (None, None)
        kupiec          : dict (POF test)
        christoffersen  : dict (CC test)
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    if not isinstance(var_series, pd.Series):
        var_series = pd.Series(var_series, index=returns.index)

    # Align & drop NaNs in VaR
    mask = ~var_series.isna()
    r = returns[mask]
    v = var_series[mask]

    if r.empty:
        raise ValueError("No overlapping observations after dropping NaNs in VaR series.")

    # A breach if realized return < VaR threshold (VaR is negative)
    breaches_bool = r < v

    n = breaches_bool.size
    x = int(breaches_bool.sum())
    exception_rate = x / n

    kupiec = _kupiec_pof(breaches_bool, alpha)
    cc = _christoffersen_cc(breaches_bool, alpha)

    if alpha == 0.99:
        zone = basel_traffic_light_99(n, x)
        thresholds = basel_thresholds_99(n)
    else:
        zone = "N/A"
        thresholds = (None, None)

    return {
        "t_obs": n,
        "n_exceptions": x,
        "exception_rate": exception_rate,
        "breaches": breaches_bool,
        "zone": zone,
        "thresholds": thresholds,
        "kupiec": kupiec,
        "christoffersen": cc,
    }
