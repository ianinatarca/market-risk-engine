# dashboard/app.py
import os
import sys
import streamlit as st

# --- Make project root importable for "utils", "risk", etc. ---
# ROOT should be the project folder
ROOT = os.path.dirname(os.path.abspath(__file__))

 

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
# ------------------------------------------------------------
# Lightweight CSS for a cleaner header
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .mims-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        margin-bottom: 0.75rem;
    }
    .mims-left {
        display: flex;
        align-items: center;
        gap: 1.25rem;
    }
    .mims-logo img {
        max-height: 70px;
    }
    .mims-title {
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0;
        padding: 0;
    }
    .mims-subtitle {
        font-size: 0.95rem;
        color: #777777;
        margin: 0.15rem 0 0 0;
    }
    .mims-right {
        text-align: right;
        font-size: 0.85rem;
        color: #777777;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Minerva / MIMS header with logo
# ------------------------------------------------------------

logo_path = os.path.join(ROOT, "minerva_logo.jpg")  

st.write("Logo path:", logo_path, "Exists:", os.path.exists(logo_path))




st.markdown("<div class='mims-header'>", unsafe_allow_html=True)

# left side: logo + title block
st.markdown("<div class='mims-left'>", unsafe_allow_html=True)
if os.path.exists(logo_path):
    st.image(logo_path, width=80)
else:
    st.markdown("**MINERVA**")

st.markdown(
    """
    <div>
        <p class="mims-title">MIMS – Multi Asset Global Opportunities Fund</p>
        <p class="mims-subtitle">
            Portfolio Management Team • Risk & Analytics Dashboard
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)  # close .mims-left

# right side: as-of date
if latest_date:
    st.markdown(
        f"""
        <div class="mims-right">
            Data as of <strong>{latest_date}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("<div class='mims-right'></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close .mims-header

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
