import numpy as np
from arch import arch_model

def garch_fit(series):
    """
    Fast GARCH(1,1)-t fit for a single return series.

    Returns:
        nu_g   : dof of t distribution
        mu_g   : 1-step ahead conditional mean
        sigma_g: 1-step ahead conditional std
    """
    y = series.dropna()

    # If too short, fall back to simple stats
    if len(y) < 50:
        return 30.0, y.mean(), y.std()

    # Scale by volatility level to avoid numerical issues
    std = y.std()
    if std < 1e-4:
        scale = 1000.0
    elif std < 1e-3:
        scale = 100.0
    else:
        scale = 1.0

    y_s = y * scale

    # FIX: no lag selection — just GARCH(1,1), which is standard
    model = arch_model(y_s, vol="GARCH", p=1, q=1, dist="t", rescale=False)
    res = model.fit(disp="off", update_freq=0, options={"maxiter": 500})

    fcst = res.forecast(reindex=False, horizon=1)
    mean_s = fcst.mean.iloc[-1, 0]
    var_s  = fcst.variance.iloc[-1, 0]

    mean_ret = mean_s / scale
    std_ret  = np.sqrt(var_s) / scale

    # get dof; fallback if missing
    nu = float(res.params.get("nu", 10.0))

    return nu, mean_ret, std_ret
