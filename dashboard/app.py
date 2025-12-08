# dashboard/app.py
import os
import sys
import streamlit as st

# --- Make project root importable for "utils", "risk", etc. ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.loaders import load_data

# ------------------------------------------------------------
# Streamlit page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Market Risk Dashboard",
    layout="wide",
    page_icon="📊"
)

# ------------------------------------------------------------
# Main Title
# ------------------------------------------------------------
st.title("📊 Real-Time Market Risk Dashboard")

st.markdown(
"""
Welcome to your **multi-asset market risk platform**.  
This dashboard integrates everything you have built so far into a unified analytical framework:

### 🔍 **What this dashboard provides**
- **Asset-level risk diagnostics**  
  Static Student-t, GARCH-t, and sensitivities (delta / DV01)

- **Portfolio risk metrics**  
  - Student-t VaR / ES  
  - Conditional GARCH-t VaR / ES  
  - Historical VaR  
  - Monte Carlo *t-copula* VaR & CVaR (1-day and 10-day horizons)

- **Stress Testing Framework**  
  COVID crash • +200 bp rate shock • Oil price spike • China slowdown  
  + per-asset attribution of losses

- **Model Validation & Backtesting**  
  Basel traffic-light classification  
  Kupiec (POF) and Christoffersen (Independence & Conditional Coverage) tests

---

Use the navigation sidebar to explore each component in detail:
1. **Portfolio Analysis** – exposures, best/worst contributors, volatility overview.  
2. **Correlations** – heatmaps and cumulative returns.  
3. **Monte Carlo Simulation** – simulated portfolio paths and risk metrics.  
4. **P&L Distributions** – histogram & CDF visualisation with VaR overlays.  
5. **Stress Tests** – immediate shocks and scenario attribution.  
6. **Backtesting** – VaR exceptions, model stability & statistical tests.  
7. **Sensitivities** – delta and DV01 breakdown (equity & rates).  
"""
)

# ------------------------------------------------------------
# Quick Portfolio Snapshot
# ------------------------------------------------------------
df_ret, w = load_data()

st.markdown("### 🧭 Portfolio Snapshot")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Assets in Portfolio",
        value=len(w[w > 0]),
        help="Number of instruments with nonzero portfolio weight."
    )

with col2:
    st.metric(
        label="Return History Length",
        value=f"{len(df_ret)} days",
        help="Number of daily observations available to estimate volatility & correlations."
    )

with col3:
    st.metric(
        label="Weight Check",
        value=f"{w.sum():.2f}",
        help="Should be equal to 1.00 for a fully invested portfolio."
    )

st.markdown(
"""
---

This dashboard is designed to help you **understand, quantify, and visualise** risk across your
portfolio in a structured, professional framework.  

Use the left-hand sidebar to continue.
"""
)
