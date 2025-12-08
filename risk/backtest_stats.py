# risk/backtest_stats.py

import numpy as np
from scipy.stats import chi2


def _to_bool_array(breaches):
    """Convert input to a clean 1D boolean numpy array."""
    b = np.asarray(breaches).astype(bool)
    if b.ndim != 1:
        b = b.ravel()
    return b


# ============================================================
# 1. Kupiec Proportion of Failures (POF) test
# ============================================================

def kupiec_pof_test(breaches, alpha=0.99):
    """
    Kupiec (1995) Proportion of Failures (POF) test.

    Parameters
    ----------
    breaches : array-like of bool
        breaches[t] = True if VaR was violated at time t.
    alpha : float
        VaR confidence level (e.g. 0.99 for 99% VaR).

    Returns
    -------
    dict with:
        n        : number of observations
        x        : number of breaches
        pi_hat   : empirical breach probability
        LR_pof   : likelihood ratio statistic
        p_value  : 1 - CDF_chi2(LR_pof, df=1)
    """
    b = _to_bool_array(breaches)
    n = b.size
    if n == 0:
        raise ValueError("No observations for Kupiec test")

    x = int(b.sum())
    pi_hat = x / n if n > 0 else 0.0

    # Guard against log(0)
    eps = 1e-10
    pi_hat_clipped = np.clip(pi_hat, eps, 1.0 - eps)

    # H0: true breach prob = (1 - alpha)
    p0 = 1.0 - alpha

    logL0 = x * np.log(p0) + (n - x) * np.log(1.0 - p0)
    logL1 = x * np.log(pi_hat_clipped) + (n - x) * np.log(1.0 - pi_hat_clipped)

    LR_pof = -2.0 * (logL0 - logL1)
    p_value = 1.0 - chi2.cdf(LR_pof, df=1)

    return {
        "n": n,
        "x": x,
        "pi_hat": pi_hat,
        "alpha": alpha,
        "LR_pof": LR_pof,
        "p_value": p_value,
    }


# ============================================================
# 2. Christoffersen (1998) Independence test
# ============================================================

def christoffersen_independence_test(breaches):
    """
    Christoffersen (1998) independence test.

    Tests whether VaR exceptions are independent over time (no clustering).

    Parameters
    ----------
    breaches : array-like of bool

    Returns
    -------
    dict with:
        n00, n01, n10, n11 : transition counts
        pi01, pi11         : estimated transition probs
        LR_ind             : LR statistic
        p_value            : p-value (df=1)
    """
    b = _to_bool_array(breaches)
    if b.size < 2:
        raise ValueError("Need at least 2 observations for independence test")

    # Build 2x2 transition counts
    # n_ij = number of transitions from state i at t-1 to j at t
    b_prev = b[:-1]
    b_curr = b[1:]

    n00 = np.sum((b_prev == 0) & (b_curr == 0))
    n01 = np.sum((b_prev == 0) & (b_curr == 1))
    n10 = np.sum((b_prev == 1) & (b_curr == 0))
    n11 = np.sum((b_prev == 1) & (b_curr == 1))

    # If there are no 0->* or 1->* transitions, the test degenerates
    if (n00 + n01 == 0) or (n10 + n11 == 0):
        return {
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "pi01": np.nan, "pi11": np.nan,
            "LR_ind": np.nan, "p_value": np.nan,
        }

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)

    # Overall unconditional exception probability
    n0 = n00 + n01
    n1 = n10 + n11
    pi = (n01 + n11) / (n0 + n1)

    # Guard against 0/1 probs in logs
    eps = 1e-10
    pi01_c = np.clip(pi01, eps, 1.0 - eps)
    pi11_c = np.clip(pi11, eps, 1.0 - eps)
    pi_c = np.clip(pi, eps, 1.0 - eps)

    # Restricted (i.i.d) log-likelihood
    logL_r = (
        n0 * np.log(1.0 - pi_c) +
        n1 * np.log(pi_c)
    )

    # Unrestricted (Markov) log-likelihood
    logL_u = (
        n00 * np.log(1.0 - pi01_c) +
        n01 * np.log(pi01_c) +
        n10 * np.log(1.0 - pi11_c) +
        n11 * np.log(pi11_c)
    )

    LR_ind = -2.0 * (logL_r - logL_u)
    p_value = 1.0 - chi2.cdf(LR_ind, df=1)

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi01": pi01, "pi11": pi11,
        "LR_ind": LR_ind,
        "p_value": p_value,
    }


# ============================================================
# 3. Christoffersen Conditional Coverage (CC) test
# ============================================================

def christoffersen_cc_test(breaches, alpha=0.99):
    """
    Christoffersen conditional coverage test:
      H0: correct unconditional coverage AND independence.

    LR_cc = LR_pof + LR_ind, df = 2.

    Parameters
    ----------
    breaches : array-like of bool
    alpha    : VaR confidence level

    Returns
    -------
    dict with:
        pof        : dict from kupiec_pof_test
        indep      : dict from christoffersen_independence_test
        LR_cc      : combined LR statistic
        p_value    : p-value (df=2)
    """
    pof = kupiec_pof_test(breaches, alpha=alpha)
    indep = christoffersen_independence_test(breaches)

    LR_pof = pof["LR_pof"]
    LR_ind = indep["LR_ind"]

    if np.isnan(LR_ind):
        LR_cc = np.nan
        p_value = np.nan
    else:
        LR_cc = LR_pof + LR_ind
        p_value = 1.0 - chi2.cdf(LR_cc, df=2)

    return {
        "pof": pof,
        "indep": indep,
        "LR_cc": LR_cc,
        "p_value": p_value,
    }


# ============================================================
# 4. Basel Traffic Light helper (99% VaR)
# ============================================================

def basel_traffic_light_99(n, x):
    """
    Basel traffic light regime for 99% VaR.

    Official thresholds are defined for 250 days:
        Green : 0–4 exceptions
        Yellow: 5–9
        Red   : >=10

    For other n, we scale those thresholds linearly by n/250.
    This is a heuristic but standard in small-sample settings.

    Parameters
    ----------
    n : int
        Number of backtesting observations
    x : int
        Number of exceptions

    Returns
    -------
    "GREEN", "YELLOW", or "RED"
    """
    if n <= 0:
        return "UNDEFINED"

    # Ratios from the official 250-day thresholds
    green_max_ratio = 4 / 250.0
    yellow_max_ratio = 9 / 250.0

    green_max = int(np.floor(green_max_ratio * n + 1e-9))
    yellow_max = int(np.floor(yellow_max_ratio * n + 1e-9))

    if x <= green_max:
        return "GREEN"
    elif x <= yellow_max:
        return "YELLOW"
    else:
        return "RED"
