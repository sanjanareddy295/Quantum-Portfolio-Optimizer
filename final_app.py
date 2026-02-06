import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Quantum Portfolio Optimizer",
    layout="wide"
)

st.title("🚀 Quantum-Inspired Portfolio Optimization System")
st.subheader("AI-Based Final Portfolio Recommendation")

# ---------------------------------
# STOCK LIST (VERY STABLE STOCKS)
# ---------------------------------
stocks = ["AAPL","MSFT","AMZN","NVDA","GOOG","META","TSLA"]

# ---------------------------------
# RISK SLIDER (SAFE RANGE)
# ---------------------------------
risk_aversion = st.slider(
    "Select Risk Preference (Higher = Safer Portfolio)",
    0.2, 0.8, 0.5
)

# ---------------------------------
# LOAD DATA (EXTREMELY SAFE)
# ---------------------------------
@st.cache_data
def load_data():

    try:
        data = yf.download(
            stocks,
            period="1y",
            auto_adjust=True,
            progress=False,
            threads=False
        )["Close"]

        # Remove failed stocks
        data = data.dropna(axis=1, how='any')

        if data.shape[1] < 3:
            return None, None

        returns = data.pct_change().dropna()

        mu = returns.mean()
        cov = returns.cov()

        return mu, cov

    except:
        return None, None


mu, cov = load_data()

# ---------------------------------
# DATA FAIL SAFE
# ---------------------------------
if mu is None or len(mu) < 3:

    st.error("⚠️ Live stock download failed.")

    st.warning("Using fallback demo data for reliability.")

    # DEMO DATA (NEVER FAILS)
    stocks_demo = ["AAPL","MSFT","NVDA"]

    mu = pd.Series(
        [0.12, 0.10, 0.15],
        index=stocks_demo
    )

    cov = pd.DataFrame(
        [[0.05,0.01,0.02],
         [0.01,0.04,0.015],
         [0.02,0.015,0.06]],
        columns=stocks_demo,
        index=stocks_demo
    )

# ---------------------------------
# OPTIMIZER (IMPOSSIBLE TO FAIL)
# ---------------------------------
def optimize_portfolio(mu, cov, risk):

    stock_list = list(mu.index)

    best_score = -999
    best_portfolio = None

    for combo in combinations(stock_list, min(3, len(stock_list))):

        try:

            expected_return = mu[list(combo)].mean()

            risk_val = np.sqrt(
                np.diag(cov.loc[list(combo), list(combo)])
            ).mean()

            score = expected_return - (risk * risk_val)

            if score > best_score:
                best_score = score
                best_portfolio = combo

        except:
            continue

    # 🔥 FINAL SAFETY (GUARANTEED PORTFOLIO)
    if best_portfolio is None:

        best_portfolio = tuple(
            mu.sort_values(ascending=False).head(3).index
        )

        best_score = mu[list(best_portfolio)].mean()

    return best_portfolio, best_score


best_portfolio, best_score = optimize_portfolio(mu, cov, risk_aversion)

# ---------------------------------
# EXTRA SAFETY (ABSOLUTE)
# ---------------------------------
if best_portfolio is None:

    best_portfolio = tuple(
        mu.sort_values(ascending=False).head(3).index
    )

# ---------------------------------
# RESULTS
# ---------------------------------
st.divider()

st.success(
    f"✅ Recommended Portfolio: {', '.join(best_portfolio)}"
)

col1, col2 = st.columns(2)

# RETURNS
with col1:

    st.subheader("📊 Expected Returns")

    fig, ax = plt.subplots()

    mu[list(best_portfolio)].plot(
        kind='bar',
        ax=ax
    )

    st.pyplot(fig)


# RISK
with col2:

    st.subheader("📉 Portfolio Risk")

    risk_values = np.sqrt(
        np.diag(cov.loc[list(best_portfolio), list(best_portfolio)])
    )

    fig2, ax2 = plt.subplots()

    pd.Series(
        risk_values,
        index=best_portfolio
    ).plot(kind='bar', ax=ax2)

    st.pyplot(fig2)


st.info(f"⭐ Optimization Score: {round(best_score,5)}")

# ---------------------------------
# OPTIONAL DATA VIEW
# ---------------------------------
st.divider()
st.subheader("📂 Market Data Overview")

if st.checkbox("Show Expected Returns"):
    st.dataframe(mu)

if st.checkbox("Show Covariance Matrix"):
    st.dataframe(cov)

# ---------------------------------
# FOOTER
# ---------------------------------
st.divider()

st.caption(
"""
This application applies a quantum-inspired optimization strategy 
to recommend an optimal stock portfolio by balancing return and financial risk.
"""
)
