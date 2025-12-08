# risk_backtest_sweep.py

from utils.loaders import load_data
from risk.backtest import rolling_historical_var, backtest_var

def main():
    df_ret, w = load_data()
    port_ret = df_ret @ w

    alpha = 0.99
    windows = [20, 30, 60, 90, 120]  # try a few

    print("\n=== Historical 99% VaR backtest sweep ===\n")

    for window in windows:
        var_hist = rolling_historical_var(port_ret, alpha=alpha, window=window)
        bt = backtest_var(port_ret, var_hist, alpha=alpha)

        print(f"Window = {window} days")
        print(f"  Observations used : {bt['t_obs']}")
        print(f"  Exceptions        : {bt['n_exceptions']}")
        print(f"  Exception rate    : {bt['exception_rate']:.3%}")
        print(f"  Basel zone        : {bt['zone']}")
        print(f"  Kupiec p-value    : {bt['kupiec']['p_value']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
