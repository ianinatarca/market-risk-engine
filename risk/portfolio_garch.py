import numpy as np
from scipy.stats import t
from .student_t import es_factor_t

def portfolio_garch_var_es(weights, mu_g, sigma_g, corr, dfs_g):
    D = np.diag(sigma_g)
    Sigma = D @ corr @ D

    mu_p = np.dot(weights, mu_g)
    sigma_p = np.sqrt(weights @ Sigma @ weights)

    nu_p = np.average(dfs_g, weights=np.abs(weights))

    VaR95 = mu_p + sigma_p * t.ppf(0.05, df=nu_p)
    VaR99 = mu_p + sigma_p * t.ppf(0.01, df=nu_p)

    ES95 = mu_p + sigma_p * es_factor_t(0.05, nu_p)
    ES99 = mu_p + sigma_p * es_factor_t(0.01, nu_p)

    return VaR95, ES95, VaR99, ES99
