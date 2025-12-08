import numpy as np
import pandas as pd
from scipy.stats import t, ks_2samp

def estimate_df_ks(series, rng, min_df=2, max_df=99):
    """Estimate degrees of freedom via KS test on standardized returns."""
    x = series.dropna()
    x_std = (x - x.mean()) / x.std()

    d = {}
    for nu in range(min_df, max_df + 1):
        sim = rng.standard_t(nu, size=len(x_std))
        d[nu] = ks_2samp(sim, x_std)[0]
    return min(d, key=d.get)

def es_factor_t(alpha, df):
    q = t.ppf(alpha, df=df)
    pdf = t.pdf(q, df=df)
    return -((df + q**2) / (df - 1)) * (pdf / alpha)

def compute_student_t_stats(df, rng):
    """Compute μ, σ, ν, VaR and ES for each asset."""
    stats = pd.DataFrame(index=df.columns)
    stats["mean"] = df.mean()
    stats["std"] = df.std(ddof=1)
    stats["df"] = np.nan

    for col in df.columns:
        stats.loc[col, "df"] = estimate_df_ks(df[col], rng)

    stats["VaR95"] = stats["mean"] + stats["std"] * t.ppf(0.05, stats["df"])
    stats["VaR99"] = stats["mean"] + stats["std"] * t.ppf(0.01, stats["df"])

    stats["ES95"] = stats["mean"] + stats["std"] * es_factor_t(0.05, stats["df"])
    stats["ES99"] = stats["mean"] + stats["std"] * es_factor_t(0.01, stats["df"])

    return stats
