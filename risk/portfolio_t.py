import numpy as np
from scipy.stats import t, ks_2samp
from .student_t import es_factor_t

def estimate_portfolio_df(port_ret, rng):
    x = port_ret.dropna()
    x_std = (x - x.mean()) / x.std()

    d = {
        nu: ks_2samp(rng.standard_t(nu, size=len(x_std)), x_std)[0]
        for nu in range(2, 100)
    }
    return min(d, key=d.get)

def portfolio_t_var_es(port_ret, nu):
    mu = port_ret.mean()
    sigma = port_ret.std(ddof=1)

    VaR95 = mu + sigma * t.ppf(0.05, df=nu)
    VaR99 = mu + sigma * t.ppf(0.01, df=nu)

    ES95 = mu + sigma * es_factor_t(0.05, nu)
    ES99 = mu + sigma * es_factor_t(0.01, nu)

    return VaR95, ES95, VaR99, ES99
