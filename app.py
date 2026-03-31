import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock trend Predictor ", layout="wide")

# =========================
# Data Fetch
# =========================
@st.cache_data
def get_data(ticker):
    df = yf.download(ticker, period="1y", auto_adjust=True)
    df.dropna(inplace=True)
    return df

# =========================
# Feature Engineering
# =========================
def add_features(df):
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['Volatility'] = df['Return'].rolling(5).std()

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    return df.dropna()

# =========================
# Model
# =========================
def train_and_predict(df):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'Volatility', 'RSI']
    
    X = df[features]
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled[:-1], y[:-1])

    pred = model.predict(X_scaled[-1].reshape(1, -1))[0]
    prob = model.predict_proba(X_scaled[-1].reshape(1, -1))[0][pred]

    return pred, prob

# =========================
# Confidence Meter
# =========================
def confidence_meter(prob):
    percent = prob * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        title={'text': "Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "green"}
            ],
        }
    ))

    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Multi-Stock Analyzer
# =========================
def analyze_multiple(tickers):
    results = []

    for ticker in tickers:
        try:
            df = get_data(ticker)
            if df is None or df.empty:
                continue
                
            df = add_features(df)
            if len(df)<50:
                continue

            pred, prob = train_and_predict(df)

            results.append({
                "Ticker": ticker,
                "Prediction": "UP" if pred == 1 else "DOWN",
                "Confidence": round(prob * 100, 2)
            })
            
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue

    return pd.DataFrame(results).sort_values(by="Confidence", ascending=False)

# =========================
# UI
# =========================
st.title("🚀 AI Stock Predictor")
st.markdown("Analyze global stocks with ML")

# main logic
tickers_input = st.text_input(
    "Enter Stock Ticker(s) (comma separated)",
    "AAPL"
)
if st.button("Predict"):
    ticker_list = list(set([t.strip().upper() for t in tickers_input.split(",")]))

    try:
        # 🔥 MULTI STOCK MODE
        if len(ticker_list) > 1:
            df_results = analyze_multiple(ticker_list)

            if df_results is None or df_results.empty:
                st.warning("No valid data found.")
            else:
                st.subheader("📊 All Stocks")
                st.dataframe(df_results)

        # 🔥 SINGLE STOCK MODE
        else:
            ticker = ticker_list[0]

            df = get_data(ticker)

            if df.empty:
                st.error(f"No data found for {ticker}")
            else:
                df = add_features(df)

                pred, prob = train_and_predict(df)

                trend = "🔼 UP" if pred == 1 else "🔽 DOWN"

                st.subheader(f"{ticker} → {trend}")

                confidence_meter(prob)
                st.line_chart(df['Close'])

    except Exception as e:
        st.error(f"Error: {e}")