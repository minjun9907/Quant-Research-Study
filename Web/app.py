import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

# 페이지 설정
st.set_page_config(page_title="Quant Research Dashboard", layout="wide")

# --- 1. 데이터 로드 (캐싱 활용) ---
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2019-01-01")
    return data

# --- 2. 사이드바 제어판 ---
st.sidebar.title("🛠 Strategy Settings")
asset = st.sidebar.selectbox("Select Asset", ["SPY", "BTC-USD"])
short_ma = st.sidebar.slider("Short MA Window", 5, 50, 10)
long_ma = st.sidebar.slider("Long MA Window", 50, 200, 100)
rsi_thresh = st.sidebar.slider("RSI Entry Threshold", 50, 90, 70)
cost = st.sidebar.number_input("Transaction Cost (%)", value=0.1) / 100

# --- 3. 전략 로직 (민준님의 연구 반영) ---
df = load_data(asset)
df['Short_MA'] = df['Close'].rolling(short_ma).mean()
df['Long_MA'] = df['Close'].rolling(long_ma).mean()

# RSI 계산
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['RSI'] = 100 - (100 / (1 + gain/loss))

# 시그널 생성 (WFA에서 검증한 로직)
df['Signal'] = 0.0
current_pos = 0
signals = []
for i in range(len(df)):
    ma_up = df['Short_MA'].iloc[i] > df['Long_MA'].iloc[i]
    rsi_ok = df['RSI'].iloc[i] < rsi_thresh
    if ma_up and (current_pos == 1 or rsi_ok):
        current_pos = 1
    else:
        current_pos = 0
    signals.append(current_pos)
df['Signal'] = signals

# 수익률 계산
df['Market_Ret'] = df['Close'].pct_change()
df['Trade'] = df['Signal'].diff().abs().fillna(0)
df['Strategy_Ret'] = (df['Market_Ret'] * df['Signal'].shift(1)) - (df['Trade'] * cost)

# --- 4. 대시보드 시각화 ---
st.title(f"📈 {asset} Momentum Strategy Dashboard")
st.markdown(f"**Description:** MA({short_ma}/{long_ma}) Crossover with RSI < {rsi_thresh} Entry Filter")

# KPI 카드
cum_ret = (1 + df['Strategy_Ret'].fillna(0)).cumprod().iloc[-1]
sharpe = (df['Strategy_Ret'].mean() / df['Strategy_Ret'].std()) * np.sqrt(252)
peak = (1 + df['Strategy_Ret'].fillna(0)).cumprod().cummax()
mdd = (((1 + df['Strategy_Ret'].fillna(0)).cumprod() / peak) - 1).min()

c1, c2, c3 = st.columns(3)
c1.metric("Cumulative Return", f"{cum_ret:.2f}x")
c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
c3.metric("Max Drawdown", f"{mdd*100:.2f}%")

# 수익률 차트
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=(1+df['Market_Ret'].fillna(0)).cumprod(), name="Market (Buy & Hold)"))
fig.add_trace(go.Scatter(x=df.index, y=(1+df['Strategy_Ret'].fillna(0)).cumprod(), name="Strategy"))
fig.update_layout(title="Strategy vs Market Performance", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)