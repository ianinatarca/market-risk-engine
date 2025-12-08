# dashboard/pages/7_Sensitivities.py

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import plotly.express as px

from utils.loaders import load_data
from risk.sensitivities import total_sensitivities, apply_equity_shock, apply_rate_shock

st.title("📊 Portfolio Sensitivities")

st.markdown(
"""
This page summarises how your portfolio reacts to **small market moves**:

- **Equity delta**: € impact for a **+1% move** in each equity.
- **DV01**: € impact for a **+1bp parallel shift** in interest rates on your bond book.

For rates we use a **simple duration-based approximation** (flat 6-year duration).
"""
)

# ------------------------------------------------------------
# 1. Load data & identify bonds
# ------------------------------------------------------------
df_ret, w = load_data()
prices = (1 + df_ret).cumprod().iloc[-1]

bond_mask = (
    prices.index.str.contains("%")
    | prices.index.str.contains("BTP")
    | prices.index.str.contains("Bund")
    | prices.index.str.contains("OAT")
    | prices.index.str.contains("ROMANIA")
)

st.caption("Bond assets detected:")
st.write(list(prices[bond_mask].index))

# ------------------------------------------------------------
# 2. Compute sensitivities
# ------------------------------------------------------------
res = total_sensitivities(w, prices, bond_mask, notional=1_000_000)

table = res["table"]
delta = res["delta"]
dv01  = res["dv01"]

# ------------------------------------------------------------
# 3. Portfolio-level summary
# ------------------------------------------------------------
st.subheader("Portfolio Summary")

col1, col2 = st.columns(2)
with col1:
    st.metric("Equity delta (€/ +1%)", f"{res['delta_total']:,.0f}")
with col2:
    st.metric("DV01 (€/ +1bp)", f"{res['dv01_total']:,.2f}")

st.markdown(
f"""
- A **+1% move in global equities** would change portfolio value by about  
  **€{res['delta_total']:,.0f}** (using current weights).
- A **+100bp parallel rate rise** would cost roughly  
  **€{apply_rate_shock(res['dv01_total'], 100):,.0f}** on the bond book.
"""
)

# ------------------------------------------------------------
# 4. Per-asset table
# ------------------------------------------------------------
st.subheader("Per-Asset Exposures")
st.dataframe(
    table.sort_values("Position €", ascending=False),
    use_container_width=True,
)

# ------------------------------------------------------------
# 5. Top equity and rate risk contributors
# ------------------------------------------------------------
# Top 10 equity risk drivers by |delta|
top_delta = delta.abs().sort_values(ascending=False).head(10)
top_delta = top_delta.reindex(top_delta.index)  # preserve order

st.subheader("Top Equity Risk Contributors (ΔPnL for +1%)")

fig_delta = px.bar(
    top_delta,
    labels={"value": "ΔPnL (€/ +1%)", "index": "Asset"},
)
st.plotly_chart(fig_delta, use_container_width=True)

# Top 10 rate risk drivers by |DV01|
top_dv01 = dv01.abs().sort_values(ascending=False).head(10)

st.subheader("Top Interest Rate Risk Contributors (|DV01|)")

fig_dv01 = px.bar(
    top_dv01,
    labels={"value": "DV01 (€/ +1bp)", "index": "Bond"},
)
st.plotly_chart(fig_dv01, use_container_width=True)

# ------------------------------------------------------------
# 6. Simple shock simulator
# ------------------------------------------------------------
st.subheader("Shock Calculator (Sensitivity-based)")

col1, col2 = st.columns(2)
with col1:
    eq_shock = st.slider("Equity shock (%)", -10, 10, 5)
with col2:
    rate_shock = st.slider("Rate shock (bps)", -200, 200, 100)

pnl_eq   = apply_equity_shock(res["delta_total"], eq_shock)
pnl_rate = apply_rate_shock(res["dv01_total"], rate_shock)

col1, col2 = st.columns(2)
with col1:
    st.metric(f"Equity shock PnL ({eq_shock:+d}%)", f"{pnl_eq:,.0f} €")
with col2:
    st.metric(f"Rate shock PnL ({rate_shock:+d} bp)", f"{pnl_rate:,.0f} €")

st.markdown(
"""
> **Note**  
> These are **first-order (linear) approximations**.  
> They are designed to give you a **clear ranking of risk drivers**, not to replace
> full pricing models.
"""
)
