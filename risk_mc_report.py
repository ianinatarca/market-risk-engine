# risk_mc_report.py
from utils.loaders import load_data         # or wherever your load_data() lives
from risk.copulas import mc_portfolio_pnl, var_cvar

def main():
    df_ret, w = load_data()


    pnl_1d = mc_portfolio_pnl(
    df_ret, w,
    notional=1_000_000,
    n_sims=100_000,
    horizon_days=1,
    lam=0.94,
    nu_copula=5,
    df_marg=5,
    cov_mode="ewma",      # <<< NOW USING EWMA
)


    var95_1d, es95_1d = var_cvar(pnl_1d, alpha=0.95)
    var99_1d, es99_1d = var_cvar(pnl_1d, alpha=0.99)

    # 10-day VaR/CVaR
    pnl_10d = mc_portfolio_pnl(df_ret, w, notional=1_000_000,
                               n_sims=100_000, horizon_days=10,
                               lam=0.94, nu_copula=5, df_marg=5)

    var95_10d, es95_10d = var_cvar(pnl_10d, alpha=0.95)
    var99_10d, es99_10d = var_cvar(pnl_10d, alpha=0.99)

    print("\n=== Monte Carlo t-copula portfolio risk (notional €1,000,000) ===")
    print("\n1-day horizon:")
    print(f"VaR 95% = {var95_1d:,.0f} €   ES 95% = {es95_1d:,.0f} €")
    print(f"VaR 99% = {var99_1d:,.0f} €   ES 99% = {es99_1d:,.0f} €")

    print("\n10-day horizon:")
    print(f"VaR 95% = {var95_10d:,.0f} €   ES 95% = {es95_10d:,.0f} €")
    print(f"VaR 99% = {var99_10d:,.0f} €   ES 99% = {es99_10d:,.0f} €")


if __name__ == "__main__":
    main()
