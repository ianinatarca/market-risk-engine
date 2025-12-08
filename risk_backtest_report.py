# risk_backtest_report.py

from utils.loaders import load_data
from risk.backtest import rolling_historical_var, backtest_var

def main():
    df_ret, w = load_data()
    port_ret = df_ret @ w

    # Backtest 99% 1-day historical VaR
    alpha = 0.99
    window = 60  # you can change this if you want a longer lookback

    var_hist = rolling_historical_var(port_ret, alpha=alpha, window=window)
    bt = backtest_var(port_ret, var_hist, alpha=alpha)

    n = bt["t_obs"]
    x = bt["n_exceptions"]
    exc_rate = bt["exception_rate"]
    zone = bt["zone"]
    green_max, yellow_max = bt["thresholds"]
    kupiec = bt["kupiec"]
    cc = bt["christoffersen"]

    expected_breaches = n * (1.0 - alpha)

    print("\n=== VaR Backtest (Historical, 1-day, 99%) ===")
    print(f"Lookback window       : {window} days")
    print(f"Observations used     : {n}")
    print(f"Expected breaches     : {expected_breaches:.2f}")
    print(f"Exceptions (breaches) : {x}")
    print(f"Exception rate        : {exc_rate:.3%}")

    if green_max is not None:
        print("\nBasel-like thresholds (approx, scaled from 250 days):")
        print(f"  GREEN  : 0–{green_max} exceptions")
        print(f"  YELLOW : {green_max+1}–{yellow_max} exceptions")
        print(f"  RED    : >= {yellow_max+1} exceptions")
    else:
        print("\nBasel traffic light   : UNDEFINED (sample too small for classification)")

    print(f"\nBasel traffic light   : {zone}")

    # Kupiec POF
    print("\n--- Kupiec POF test (unconditional coverage) ---")
    print(f"pi_hat (empirical breach prob) : {kupiec['pi_hat']:.4f}")
    print(f"LR_pof                         : {kupiec['LR_pof']:.4f}")
    print(f"p-value                        : {kupiec['p_value']:.4f}")

    # Christoffersen CC
    print("\n--- Christoffersen Conditional Coverage test ---")
    print(f"LR_cc                          : {cc['LR_cc']:.4f}")
    print(f"p-value                        : {cc['p_value']:.4f}")

if __name__ == "__main__":
    main()
