import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd

from utils.loaders import load_data
from risk.stress import run_all_stress

st.title("🧨 Stress Tests")

st.markdown(
"""
Here we shock the portfolio with **pre-defined macro scenarios**:

- **COVID Crash** – equity drawdown, flight to quality  
- **+200bps Rate Shock** – parallel upward move in yields  
- **Oil Spike** – energy rally, growth under pressure  
- **China Slowdown** – EM / commodities hit

Each scenario scales asset returns using intuitive stress factors and reports the
**immediate PnL impact** and **top / bottom contributors**.
"""
)

df_ret, w = load_data()
notional = st.number_input("Portfolio notional (€)", 100_000, 5_000_000, 1_000_000, step=50_000)

if st.button("Run all scenarios"):
    results = run_all_stress(df_ret, w, notional=notional)

    # Summary table
    summary = pd.DataFrame({
        "Scenario": list(results.keys()),
        "Portfolio PnL (€)": [res["portfolio_pnl"] for res in results.values()],
    }).set_index("Scenario")

    st.subheader("Scenario comparison – immediate impact")
    st.dataframe(summary.style.format({"Portfolio PnL (€)": "{:,.0f}"}))

    # Details per scenario
    for scen, res in results.items():
        with st.expander(f"{scen} – details", expanded=False):
            st.markdown(f"**Portfolio PnL:** {res['portfolio_pnl']:,.0f} €")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Worst 5 assets (loss)**")
                st.dataframe(res["worst5"].rename("PnL €").to_frame().style.format("{:,.0f}"))
            with c2:
                st.markdown("**Best 5 assets (gain/hedge)**")
                st.dataframe(res["best5"].rename("PnL €").to_frame().style.format("{:,.0f}"))

    st.markdown(
    """
    **Use-case:** this page answers *“what if markets move violently tomorrow?”* and
    highlights which names **amplify** or **hedge** those shocks.
    """
    )
