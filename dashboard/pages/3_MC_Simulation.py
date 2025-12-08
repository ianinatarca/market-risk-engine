import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import plotly.express as px
import numpy as np

from utils.loaders import load_data
from risk.copulas import mc_portfolio_pnl, var_cvar

st.title("🎲 Monte Carlo t-Copula VaR / ES")

st.markdown(
    """
This page simulates **correlated market moves** for your portfolio using a
**t-copula Monte Carlo engine**:

- Volatilities & correlations from **EWMA** of your returns  
- Dependence across assets via a **Student-t copula** (fat-tail, crashy)  
- 1-day and 10-day **VaR / ES** on your actual portfolio weights
"""
)

# 1. Load data
df_ret, w = load_data()

# 2. Controls
col_left, col_right = st.columns(2)
with col_left:
    notional = st.number_input(
        "Portfolio notional (€)",
        min_value=100_000,
        max_value=5_000_000,
        value=1_000_000,
        step=50_000,
        format="%i",
    )
    horizon = st.selectbox("Horizon (days)", [1, 10], index=0)

with col_right:
    n_sims = st.number_input(
        "Number of simulations",
        min_value=5_000,
        max_value=200_000,
        value=50_000,
        step=5_000,
        format="%i",
    )
    lam = st.slider("EWMA λ (decay)", 0.80, 0.99, 0.94, step=0.01)

if st.button("Run simulation"):
    pnl = mc_portfolio_pnl(
        df_ret, w,
        notional=notional,
        n_sims=n_sims,
        horizon_days=horizon,
        lam=lam,
        nu_copula=5,
        df_marg=5,
    )

    # 3. Compute risk measures
    var95, es95 = var_cvar(pnl, alpha=0.95)
    var99, es99 = var_cvar(pnl, alpha=0.99)

    # Express in % of notional
    var95_pct = var95 / notional
    es95_pct  = es95  / notional
    var99_pct = var99 / notional
    es99_pct  = es99  / notional

    st.subheader("Risk metrics")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("VaR 95% (€)", f"{var95:,.0f}")
        st.metric("ES 95% (€)",  f"{es95:,.0f}")
    with col2:
        st.metric("VaR 95% (% of notional)", f"{var95_pct:.2%}")
        st.metric("ES 95% (% of notional)",  f"{es95_pct:.2%}")

    col3, col4 = st.columns(2)
    with col3:
        st.metric("VaR 99% (€)", f"{var99:,.0f}")
        st.metric("ES 99% (€)",  f"{es99:,.0f}")
    with col4:
        st.metric("VaR 99% (% of notional)", f"{var99_pct:.2%}")
        st.metric("ES 99% (% of notional)",  f"{es99_pct:.2%}")

    # 4. P&L distribution plot
    st.subheader("Simulated P&L distribution")
    fig = px.histogram(pnl, nbins=60, labels={"value": "PnL (€)"})
    st.plotly_chart(fig, use_container_width=True)

# 5. Short explanation under the chart

st.markdown(
    f"""
**How to interpret these numbers**

- **VaR 95% = {var95:,.0f} €**  
  → In 95% of simulated scenarios, the loss is **smaller** (less severe) than this number.  
  → Only **5% of scenarios** exceed this loss.

- **ES 95% = {es95:,.0f} €**  
  → The **average loss** in the worst 5% of simulated scenarios.  
  → This tells you how large losses tend to be when the market is stressed.

- **VaR 99% and ES 99%**  
  → These measure risk in the **extreme 1% tail**.  
  → They represent rarer but more severe market shocks and therefore show larger losses.
"""
)
