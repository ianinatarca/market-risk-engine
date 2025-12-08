

import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st
import plotly.express as px
import pandas as pd

from utils.loaders import load_data

st.title("🔗 Correlations & Regime Overview")

# --------------------------------------------------------
# Load data
# --------------------------------------------------------
df_ret, w = load_data()
corr = df_ret.corr()
cum_ret = (1 + df_ret).cumprod() - 1

st.markdown(
"""
This page answers two questions:

1. **How do assets move together?** (correlation matrix)  
2. **What regime are we in?** (historical cumulative returns)

Correlations are computed on **daily log-returns** over your current data window.
"""
)

# --------------------------------------------------------
# Quick summary stats
# --------------------------------------------------------
abs_corr = corr.where(~corr.isna()).abs()

import numpy as np  # <-- add this at the top






import numpy as np

upper = abs_corr.where(~np.tril(np.ones_like(abs_corr), k=0).astype(bool))

avg_abs = upper.stack().mean()
max_corr = upper.stack().max()
min_corr = corr.where(~np.tril(np.ones_like(corr), k=0).astype(bool)).stack().min()




c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Avg |correlation|", f"{avg_abs:.2f}")
with c2:
    st.metric("Max correlation", f"{max_corr:.2f}")
with c3:
    st.metric("Min correlation", f"{min_corr:.2f}")

# --------------------------------------------------------
# Correlation heatmap
# --------------------------------------------------------
st.subheader("Correlation Matrix (daily returns)")
fig = px.imshow(
    corr,
    x=corr.columns,
    y=corr.index,
    aspect="auto",
    color_continuous_midpoint=0,
    color_continuous_scale="RdBu_r",
)
fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)




# Top / bottom correlated pairs
pairs = (
    corr.where(np.tril(np.ones_like(corr), k=0).astype(bool))
    .stack()
    .reset_index()
)
pairs.columns = ["Asset 1", "Asset 2", "Correlation"]

# ---- EXCLUDE self-correlations ----
pairs = pairs[pairs["Asset 1"] != pairs["Asset 2"]]

# Top / bottom
top5 = pairs.sort_values("Correlation", ascending=False).head(5)
bottom5 = pairs.sort_values("Correlation", ascending=True).head(5)

c4, c5 = st.columns(2)
with c4:
    st.markdown("**Most positively correlated pairs**")
    st.dataframe(top5.style.format({"Correlation": "{:.2f}"}))
with c5:
    st.markdown("**Most diversifying / negatively correlated pairs**")
    st.dataframe(bottom5.style.format({"Correlation": "{:.2f}"}))





# --------------------------------------------------------
# Cumulative returns
# --------------------------------------------------------



st.subheader("Cumulative Returns (normalised to 0%)")
assets = st.multiselect(
    "Select assets to display",
    options=df_ret.columns,
    default=df_ret.columns[:5]  # show only first 5 initially
)

fig2 = px.line(
    cum_ret[assets],
    labels={"value": "Cumulative return", "index": "Date", "variable": "Asset"},
)

st.plotly_chart(fig2, use_container_width=True)





st.markdown(
"""
**Reading this page**

- Dark red / dark blue blocks in the heatmap show **clusters of risk** and **diversifiers**.  
- The cumulative return chart lets you quickly see **winners vs laggards** and any
  recent **regime shifts** (e.g. equity sell-off, bond rally).
"""
)
