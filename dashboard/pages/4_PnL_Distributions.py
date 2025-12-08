import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

from utils.loaders import load_data
from risk.copulas import mc_portfolio_pnl, var_cvar

st.title("📉 P&L Distributions – 1-Day vs 10-Day")

df_ret, w = load_data()

st.markdown(
"""
Here you can **visualise uncertainty**:

- Simulated **1-day** and **10-day** PnL distributions  
- VaR / CVaR overlays  
- Summary statistics

The simulations use the same **t-copula Monte Carlo** engine as the previous page.
"""
)

notional = st.number_input("Portfolio notional (€)", 100_000, 5_000_000, 1_000_000, step=50_000)
n_sims   = st.number_input("Number of simulations", 10_000, 200_000, 50_000, step=5_000)

if st.button("Run simulations"):
    pnl_1d  = mc_portfolio_pnl(df_ret, w, notional=notional, n_sims=n_sims, horizon_days=1)
    pnl_10d = mc_portfolio_pnl(df_ret, w, notional=notional, n_sims=n_sims, horizon_days=10)

    var95_1, es95_1 = var_cvar(pnl_1d, alpha=0.95)
    var99_1, es99_1 = var_cvar(pnl_1d, alpha=0.99)
    var95_10, es95_10 = var_cvar(pnl_10d, alpha=0.95)
    var99_10, es99_10 = var_cvar(pnl_10d, alpha=0.99)

    # ---- Plots ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1-Day P&L")
        fig1 = px.histogram(pnl_1d, nbins=60, labels={"value": "PnL (€)"})
        for q, v in [("VaR 95%", var95_1), ("VaR 99%", var99_1)]:
            fig1.add_vline(x=v, line_dash="dash", annotation_text=q, annotation_position="top")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("10-Day P&L")
        fig2 = px.histogram(pnl_10d, nbins=60, labels={"value": "PnL (€)"})
        for q, v in [("VaR 95%", var95_10), ("VaR 99%", var99_10)]:
            fig2.add_vline(x=v, line_dash="dash", annotation_text=q, annotation_position="top")
        st.plotly_chart(fig2, use_container_width=True)

    # ---- Summary table ----
    summary = pd.DataFrame({
        "Horizon": ["1-day", "1-day", "10-day", "10-day"],
        "Level":   ["95%",   "99%",   "95%",    "99%"],
        "VaR (€)": [var95_1, var99_1, var95_10, var99_10],
        "CVaR (€)": [es95_1, es99_1, es95_10, es99_10],
        "Mean (€)": [np.mean(pnl_1d), np.mean(pnl_1d),
                     np.mean(pnl_10d), np.mean(pnl_10d)],
        "Std dev (€)": [np.std(pnl_1d), np.std(pnl_1d),
                       np.std(pnl_10d), np.std(pnl_10d)],
    })

    st.subheader("Risk Metrics Summary")
    st.dataframe(
        summary.set_index(["Horizon", "Level"])
               .style.format("{:,.0f}")
    )
st.markdown(
    f"""
**How to read the table**

- Each row is a **horizon / confidence level** pair  
  – e.g. *1-day, 95%* vs *10-day, 99%*.

- **VaR (€)**  
  - For a 1-day 95% row, VaR is the loss level that is exceeded in only **5% of simulations**.  
  - For 99%, it’s exceeded in only **1% of simulations**.  
  - Larger (more negative) VaR ⇒ worse potential loss at that horizon.

- **CVaR (€)** (Expected Shortfall)  
  - The **average loss** *given* that you are already in the tail beyond VaR.  
  - It tells you: *“On very bad days (worst 5% or 1%), what is the typical loss?”*

- **Mean (€)**  
  - The average simulated P&L over the horizon.  
  - It can be slightly positive or negative depending on the sample of historical returns,  
    but it is **not** the main risk measure here.

- **Std dev (€)**  
  - The simulated volatility of P&L over that horizon.  
  - Higher values mean more dispersion of outcomes.

**Why 10-day risk is not just √10 times 1-day risk**

In a simple Gaussian world with constant volatility, you would expect  
10-day risk ≈ 1-day risk × √10.  
Here it doesn’t hold exactly because:

- Volatility is **time-varying**, estimated via EWMA/GARCH-style dynamics.  
- Returns are **fat-tailed** and **correlated** through the t-copula, which amplifies
  multi-day tail events.  

As a result, 10-day VaR / CVaR can grow **more or less than √10**, reflecting the
combined effect of stochastic volatility, heavy tails, and dependence across assets.
"""
)

