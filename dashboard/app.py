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
    page_title="MIMS – Multi Asset Global Opportunities Fund – Risk Dashboard",
    layout="wide",
    page_icon="🦁",
)

# ------------------------------------------------------------
# Load data once for snapshot
# ------------------------------------------------------------
df_ret, w = load_data()
latest_date = df_ret.index.max().date() if hasattr(df_ret.index, "max") else None

# ------------------------------------------------------------
# Minerva / MIMS header with logo
# ------------------------------------------------------------
header_col1, header_col2 = st.columns([1, 3])

with header_col1:
    # Adjust this path to wherever you saved the Minerva logo
    logo_path = os.path.join(ROOT, "data", "minerva_logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_column_width=False, width=120)
    else:
        st.markdown("### MINERVA\nInvestment Management Society")

with header_col2:
    st.markdown(
        """
        ### Minerva Investment Management Society  
        **MIMS – Multi Asset Global Opportunities Fund**  
        *Portfolio Management Team Risk Dashboard*
        """
    )
    if latest_date:
        st.caption(f"Data as of {latest_date}")

st.markdown("---")

# ------------------------------------------------------------
# High-level description
# ------------------------------------------------------------
st.markdown(
    """
This application is the **internal risk and analytics dashboard** for the  
**MIMS – Multi Asset Global Opportunities Fund**, a virtual multi-asset portfolio
managed by Minerva Investment Management Society (MIMS).

It consolidates the main tools used by the Portfolio Management Team:

- **Asset-level risk diagnostics**  
  Static Student-t, GARCH-t and sensitivities (delta / DV01).

- **Fund-level risk metrics**  
  - Student-t VaR / ES  
  - Conditional GARCH-t VaR / ES  
  - Historical VaR  
  - Monte Carlo *t-copula* VaR & CVaR (1-day and 10-day horizons)

- **Stress-testing framework**  
  COVID crash • +200 bps rates • Oil spike • China slowdown  
  with per-asset attribution of losses.

- **Model validation & backtesting**  
  Basel-style traffic-light view, Kupiec and Christoffersen tests.

Use the sidebar to move between modules. This home page gives a **quick snapshot**
of the current fund profile.
"""
)

# ------------------------------------------------------------
# Quick Fund Snapshot
# ------------------------------------------------------------
st.markdown("### 🧭 Fund Snapshot")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Number of holdings",
        value=int((w[w > 0]).shape[0]),
        help="Count of instruments with a strictly positive weight in the fund.",
    )

with col2:
    st.metric(
        label="Return history",
        value=f"{len(df_ret)} trading days",
        help="Length of the daily time series used to estimate volatility and correlations.",
    )

with col3:
    st.metric(
        label="Net exposure",
        value=f"{w.sum():.1%}",
        help="Sum of portfolio weights. Should be close to 100% for a fully invested, long-only fund.",
    )

# ------------------------------------------------------------
# Mandate / investment approach blurb (Minerva-flavoured)
# ------------------------------------------------------------
st.markdown(
    """
### 🧾 Investment Mandate & Process (summary)

- **Mandate**  
  The MIMS Multi Asset Global Opportunities Fund is a diversified, long-only
  portfolio investing in **listed equities, fixed income and selected alternatives**.
  The objective is to deliver attractive long-term total returns while controlling
  drawdowns and overall risk.

- **Top-down allocation**  
  Asset-class and regional weights are driven by Minerva’s macro and valuation views
  (growth, inflation, rates, risk premia).

- **Bottom-up selection**  
  Within each sleeve, securities are chosen based on fundamentals, quality of
  balance sheet, cash-flow resilience and valuation.

- **Risk management**  
  Derivatives, where used, are for **hedging and risk management only** (no net
  leverage). Portfolio construction is guided by **marginal risk contribution** and
  cross-asset diversification.
"""
)

# ------------------------------------------------------------
# Navigation hint
# ------------------------------------------------------------
st.markdown(
    """
---

### 📂 How to use the dashboard

1. **Portfolio Analysis** – exposures, top/bottom risk contributors, volatility overview.  
2. **Correlations** – correlation heatmaps and cumulative return profiles.  
3. **Monte Carlo Simulation** – simulated portfolio P&L paths and VaR / ES.  
4. **P&L Distributions** – 1-day vs 10-day P&L histograms with VaR overlays.  
5. **Stress Tests** – scenario losses and per-asset attribution.  
6. **Backtesting** – VaR exceptions, Basel zones, Kupiec & Christoffersen tests.  
7. **Sensitivities** – equity delta and rates DV01 breakdown.

Use the left-hand sidebar to access each section.
"""
)

# ------------------------------------------------------------
# Disclaimer (small text, Minerva-style)
# ------------------------------------------------------------
st.markdown(
    """
<hr style="margin-top:2rem;margin-bottom:0.5rem;">

<small>
<strong>Disclaimer</strong> – This dashboard is part of an academic project for the
Minerva Investment Management Society. It does not constitute investment advice,
an offer, or a solicitation to buy or sell any security or to adopt any investment
strategy. All figures are based on historical data, modelling assumptions and
simulated scenarios and are for illustrative purposes only. They should not be
interpreted as recommendations. Any use of this information is at the sole risk
and discretion of the reader.
<br>
&copy; Minerva Investment Management Society 2025. All rights reserved.
</small>
""",
    unsafe_allow_html=True,
)
