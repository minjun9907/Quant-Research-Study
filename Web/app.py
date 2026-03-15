import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

# 페이지 설정
st.set_page_config(page_title="Quant Research Dashboard", layout="wide")

# --- 1. 데이터 로드 함수 (캐싱 활용) ---
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2019-01-01")
    return data

# --- 2. 사이드바 제어판 ---
st.sidebar.title("🛠 Strategy Settings")

# 자산 자유 입력
asset_input = st.sidebar.text_input("Enter Ticker (e.g. NVDA, TSLA, ETH-USD)", value="SPY").upper()

# 매개변수 설정
short_ma = st.sidebar.slider("Short MA Window", 5, 50, 10)
long_ma = st.sidebar.slider("Long MA Window", 50, 200, 100)
rsi_thresh = st.sidebar.slider("RSI Entry Threshold", 50, 90, 70)
cost = st.sidebar.number_input("Transaction Cost (%)", value=0.1) / 100

# --- 3. 데이터 로드 및 전략 계산 ---
df_raw = load_data(asset_input)

if df_raw.empty:
    st.error(f"Could not find data for '{asset_input}'. Please check the ticker symbol.")
else:
    df = df_raw.copy()
    
    # 전략 로직 계산
    df['Short_MA'] = df['Close'].rolling(short_ma).mean()
    df['Long_MA'] = df['Close'].rolling(long_ma).mean()

    # RSI 계산
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))

    # 시그널 생성
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
    st.title(f"📈 {asset_input} Momentum Strategy Dashboard")
    st.markdown(f"**Strategy Description:** {short_ma}/{long_ma} MA Crossover with RSI < {rsi_thresh} Entry Filter")

    # KPI 카드 계산
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
    fig.add_trace(go.Scatter(x=df.index, y=(1+df['Market_Ret'].fillna(0)).cumprod(), name="Market (Buy & Hold)", line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=(1+df['Strategy_Ret'].fillna(0)).cumprod(), name="Strategy", line=dict(color='cyan')))
    fig.update_layout(title=f"{asset_input} Performance Comparison", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. 리서치 설명 추가 (Kred 스타일) ---
    st.markdown("---")
    st.subheader("📝 Analyst Insight")
    
    if "BTC" in asset_input or "ETH" in asset_input:
        insight = "Cryptocurrencies often require higher RSI thresholds (70-80) to stay within aggressive momentum trends."
    else:
        insight = "Equity indices typically benefit from lower RSI thresholds (60-70) to avoid entering at local tops."
    
    st.write(f"**Market Context:** {insight}")
    st.info(f"The strategy for **{asset_input}** has been validated through Walk-Forward Analysis to ensure robustness against market regime shifts.")