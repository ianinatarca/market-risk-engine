import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import plotly.express as px
import pandas as pd

from utils.loaders import load_data
from risk.backtest import rolling_historical_var, backtest_var


# --------------------------------------------------------
# Title & Intro
# --------------------------------------------------------
st.title("🧪 VaR Backtesting & Model Validation")

st.markdown(
"""
Backtesting compares **predicted VaR** with **realised PnL**:

- Breaches indicate when realised losses exceed VaR  
- The pattern and frequency of breaches tell you how the model behaves in the tails  
- Kupiec and Christoffersen tests provide formal diagnostics  
"""
)


# --------------------------------------------------------
# Load data
# --------------------------------------------------------
df_ret, w = load_data()
port_ret = df_ret @ w


# --------------------------------------------------------
# User inputs
# --------------------------------------------------------
alpha  = st.selectbox("Confidence level", [0.95, 0.99], index=1)
window = st.slider("Historical VaR lookback window (days)", 20, 120, 60, step=10)


# --------------------------------------------------------
# Compute VaR and backtest stats
# --------------------------------------------------------
var_hist = rolling_historical_var(port_ret, alpha=alpha, window=window)
bt = backtest_var(port_ret, var_hist, alpha=alpha)

# Align returns and VaR for plotting
mask = ~var_hist.isna()
r = port_ret[mask]
v = var_hist[mask]
breaches = bt["breaches"]


# --------------------------------------------------------
# Smooth coverage classification (no Basel)
# --------------------------------------------------------
expected_rate = 1 - alpha
exception_rate = bt["exception_rate"]
diff = exception_rate - expected_rate

if abs(diff) < 0.5 * expected_rate:
    coverage_label = "in line with expected coverage"
elif diff > 0:
    coverage_label = "higher breach frequency (under-coverage)"
else:
    coverage_label = "lower breach frequency (conservative)"


# --------------------------------------------------------
# Summary metrics
# --------------------------------------------------------
st.subheader("Backtest summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Observations", bt["t_obs"])

with col2:
    st.metric("Exceptions", bt["n_exceptions"])

with col3:
    st.metric("Exception rate", f"{exception_rate:.3%}")

with col4:
    st.metric("Coverage assessment", coverage_label)

st.markdown(
    f"""
- **Expected breach rate**: {expected_rate:.2%}  
- **Actual breach rate**: {exception_rate:.2%}
"""
)


# --------------------------------------------------------
# Kupiec & Christoffersen tests
# --------------------------------------------------------
kupiec = bt["kupiec"]
cc = bt["christoffersen"]

c5, c6 = st.columns(2)

with c5:
    st.markdown("### Kupiec POF (unconditional coverage)")
    st.write(f"Empirical π̂: {kupiec['pi_hat']:.4f}")
    st.write(f"LRᵖᵒᶠ: {kupiec['LR_pof']:.4f}")
    st.write(f"p-value: {kupiec['p_value']:.4f}")

with c6:
    st.markdown("### Christoffersen conditional coverage")
    st.write(f"LRᶜᶜ: {cc['LR_cc']:.4f}")
    st.write(f"p-value: {cc['p_value']:.4f}")


# --------------------------------------------------------
# Plot returns vs VaR and highlight breaches
# --------------------------------------------------------
st.subheader("Returns vs VaR (exceptions highlighted)")

df_plot = pd.DataFrame({
    "Return": r,
    "VaR": v,
    "Breach": breaches,
})

fig = px.line(
    df_plot,
    y="Return",
    labels={"index": "Date", "value": "Return"},
)

# Add VaR line
fig.add_scatter(
    x=df_plot.index,
    y=df_plot["VaR"],
    mode="lines",
    name="VaR",
)

# Highlight breaches
breach_points = df_plot[df_plot["Breach"]]
if not breach_points.empty:
    fig.add_scatter(
        x=breach_points.index,
        y=breach_points["Return"],
        mode="markers",
        marker=dict(size=8),
        name="VaR breach",
    )

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------
# Interpretation (neutral)
# --------------------------------------------------------
st.markdown(
f"""
### How to interpret this

- Compare the **actual breach rate** with the **expected** {expected_rate:.2%}.  
- The **coverage assessment** summarises whether deviations are small, conservative,
  or suggest under-coverage.  
- **Kupiec** tests unconditional frequency of breaches.  
- **Christoffersen** tests both frequency and independence of breaches.
"""
)
