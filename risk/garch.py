import numpy as np
from arch import arch_model
from scipy.stats import t
from .student_t import es_factor_t


#def select_lags(series, p_max=5, q_max=5):
 #   series = series.dropna()
   # best_bic = np.inf
    #best_p, best_q = 1, 1

   # for p in range(1, p_max + 1):
    #    for q in range(1, q_max + 1):
     #       try:
       #         model = arch_model(series, vol="Garch", p=p, q=q, dist="t")
        #        res = model.fit(disp="off")
         #       if res.bic < best_bic:
          #          best_bic = res.bic
          #          best_p, best_q = p, q
         #   except:
          #      continue
  #  return best_p, best_q
def select_lags(series, p_max=5, q_max=5):
    y = series.dropna()
    best_bic = np.inf
    best_p, best_q = 1, 1

    for p in range(1, p_max + 1):
        for q in range(1, q_max + 1):
            try:
                model = arch_model(y, vol="GARCH", p=p, q=q, dist="t", rescale=False)
                res = model.fit(disp="off")
                if res.bic < best_bic:
                    best_bic = res.bic
                    best_p, best_q = p, q
            except Exception:
                continue

    return best_p, best_q




 

def garch_fit(series):
    y = series.dropna()

    # choose a scaling based on volatility level
    std = y.std()
    if std < 1e-4:        # super tiny (bonds)
        scale = 1000.0
    elif std < 1e-3:      # small
        scale = 100.0
    else:
        scale = 1.0

    y_s = y * scale

    # lag selection on scaled series
    p, q = select_lags(y_s)

    # IMPORTANT: rescale=False because we already scaled manually
    model = arch_model(y_s, vol="GARCH", p=p, q=q, dist="t", rescale=False)
    res = model.fit(disp="off", options={"maxiter": 2000})

    fcst = res.forecast(reindex=False, horizon=1)

    mean_s = fcst.mean.iloc[-1, 0]
    var_s  = fcst.variance.iloc[-1, 0]

    # scale back to original units
    mean_ret = mean_s / scale
    std_ret  = np.sqrt(var_s) / scale

    return res.params["nu"], mean_ret, std_ret


#*def garch_fit(series):
   # series_scaled = series.dropna() * 1000

    #p, q = select_lags(series_scaled)

    #model = arch_model(series_scaled, vol="Garch", p=p, q=q, dist="t")
    #res = model.fit(disp="off")

    #fcst = res.forecast(reindex=False)
    
    #mean_ret = fcst.mean.iloc[0, 0] / 1000
    #std_ret  = np.sqrt(fcst.variance.iloc[0, 0]) / 1000

    #return res.params["nu"], mean_ret, std_ret*#