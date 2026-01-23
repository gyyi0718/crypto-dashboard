# streamlit_app.py
# -*- coding: utf-8 -*-
"""
Crypto Trading Dashboard
- Bybit API 사용 (rate limit 높음, 빠름)
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================
# 페이지 설정
# ==============================
st.set_page_config(
    page_title="Crypto Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 8px;
    }
    .profit { color: #43e97b !important; }
    .loss { color: #f5576c !important; }
    .signal-box {
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# 설정
# ==============================
SYMBOLS = {
    "BTCUSDT": {"name": "Bitcoin", "icon": "₿"},
    "ETHUSDT": {"name": "Ethereum", "icon": "Ξ"},
    "SOLUSDT": {"name": "Solana", "icon": "◎"},
    "XRPUSDT": {"name": "XRP", "icon": "✕"},
    "DOGEUSDT": {"name": "Dogecoin", "icon": "Ð"},
    "BNBUSDT": {"name": "BNB", "icon": "🔶"},
}

SYMBOL_LIST = list(SYMBOLS.keys())

# ==============================
# Bybit API
# ==============================

@st.cache_data(ttl=10)
def get_all_tickers():
    """Bybit 티커 조회"""
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        params = {"category": "spot"}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        if data.get("retCode") != 0:
            return {}
        
        tickers = {}
        for item in data.get("result", {}).get("list", []):
            symbol = item.get("symbol")
            if symbol in SYMBOLS:
                tickers[symbol] = {
                    'price': float(item.get("lastPrice", 0)),
                    'change': float(item.get("price24hPcnt", 0)) * 100,
                    'high': float(item.get("highPrice24h", 0)),
                    'low': float(item.get("lowPrice24h", 0)),
                    'volume': float(item.get("turnover24h", 0)),
                }
        return tickers
    except Exception as e:
        return {}


@st.cache_data(ttl=10)
def fetch_klines(symbol, interval="1", limit=200):
    """Bybit 캔들 데이터"""
    try:
        url = "https://api.bybit.com/v5/market/kline"
        params = {"category": "spot", "symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        if data.get("retCode") != 0:
            return None
        
        klines = data.get("result", {}).get("list", [])
        if not klines:
            return None
        
        df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        df = df.sort_values("timestamp").set_index("timestamp")
        return df[["open", "high", "low", "close", "volume"]]
    except:
        return None


# ==============================
# 기술적 지표
# ==============================

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).fillna(50)


def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd, macd.ewm(span=signal).mean(), macd - macd.ewm(span=signal).mean()


def calculate_bollinger(prices, period=20, std=2):
    sma = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    return sma + (std_dev * std), sma, sma - (std_dev * std)


def get_trading_signal(df):
    if df is None or len(df) < 30:
        return "HOLD", 0.5, {'rsi': 50, 'macd_cross': 0, 'ema_trend': 'N/A', 'score': 0}
    
    close = df['close']
    rsi = calculate_rsi(close).iloc[-1]
    macd, macd_signal, _ = calculate_macd(close)
    macd_cross = macd.iloc[-1] - macd_signal.iloc[-1]
    if pd.isna(macd_cross): macd_cross = 0
    
    ema_short = close.ewm(span=10).mean().iloc[-1]
    ema_long = close.ewm(span=30).mean().iloc[-1]
    ema_trend = "UP" if ema_short > ema_long else "DOWN"
    
    score = 0
    if rsi < 30: score += 2
    elif rsi > 70: score -= 2
    elif rsi < 45: score += 1
    elif rsi > 55: score -= 1
    if macd_cross > 0: score += 1
    else: score -= 1
    if ema_trend == "UP": score += 1
    else: score -= 1
    
    if score >= 2: return "LONG", min(0.5 + score * 0.1, 0.9), {'rsi': rsi, 'macd_cross': macd_cross, 'ema_trend': ema_trend, 'score': score}
    elif score <= -2: return "SHORT", min(0.5 + abs(score) * 0.1, 0.9), {'rsi': rsi, 'macd_cross': macd_cross, 'ema_trend': ema_trend, 'score': score}
    return "HOLD", 0.5, {'rsi': rsi, 'macd_cross': macd_cross, 'ema_trend': ema_trend, 'score': score}


# ==============================
# UI
# ==============================

st.title("📈 Crypto Trading Dashboard")
st.caption("Real-time analysis • BTC, ETH, SOL, XRP, DOGE, BNB")

# 사이드바
st.sidebar.title("⚙️ Settings")
selected_symbol = st.sidebar.selectbox("📌 Symbol", SYMBOL_LIST,
    format_func=lambda x: f"{SYMBOLS[x]['icon']} {SYMBOLS[x]['name']}")
interval_map = {"1분": "1", "5분": "5", "15분": "15", "1시간": "60", "4시간": "240"}
interval = interval_map[st.sidebar.selectbox("⏱️ Timeframe", list(interval_map.keys()))]
show_indicators = st.sidebar.checkbox("📊 Indicators", value=True)
auto_refresh = st.sidebar.checkbox("🔄 Auto (10s)", value=False)
if st.sidebar.button("🔄 Refresh"): st.cache_data.clear(); st.rerun()

# 전체 현황
st.subheader("🌐 Market Overview")
tickers = get_all_tickers()

if tickers:
    cols = st.columns(6)
    for i, sym in enumerate(SYMBOL_LIST):
        if sym in tickers:
            d = tickers[sym]
            with cols[i]:
                chg = d['change']
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:20px">{SYMBOLS[sym]['icon']}</div>
                    <div style="font-size:11px;color:#888">{SYMBOLS[sym]['name']}</div>
                    <div style="font-size:16px;font-weight:bold;color:#fff">${d['price']:,.4f}</div>
                    <div class="{'profit' if chg>=0 else 'loss'}">{'▲' if chg>=0 else '▼'} {chg:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.warning("⏳ Loading...")

st.divider()

# 상세
info = SYMBOLS[selected_symbol]
st.subheader(f"{info['icon']} {info['name']} Analysis")

df = fetch_klines(selected_symbol, interval=interval)
price = tickers.get(selected_symbol, {}).get('price', 0) if tickers else 0

if df is not None and not df.empty:
    signal, conf, ind = get_trading_signal(df)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Price", f"${price:,.4f}", f"{tickers.get(selected_symbol,{}).get('change',0):+.2f}%" if tickers else "")
    c2.metric(f"{'🟢' if signal=='LONG' else '🔴' if signal=='SHORT' else '⚪'} Signal", signal, f"{conf*100:.0f}%")
    c3.metric("📊 RSI", f"{ind['rsi']:.1f}", "Oversold" if ind['rsi']<30 else "Overbought" if ind['rsi']>70 else "Neutral")
    c4.metric("📈 Trend", ind['ema_trend'])
    vol = tickers.get(selected_symbol,{}).get('volume',0) if tickers else 0
    c5.metric("📊 Volume", f"${vol/1e6:.1f}M" if vol<1e9 else f"${vol/1e9:.1f}B")
    
    st.divider()
    col_chart, col_ind = st.columns([2, 1])
    
    with col_chart:
        rows, heights = (3, [0.5,0.25,0.25]) if show_indicators else (2, [0.7,0.3])
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=heights)
        
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color='#43e97b', decreasing_line_color='#f5576c'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['close'].ewm(span=10).mean(), name='EMA10', line=dict(color='#4facfe',width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['close'].ewm(span=30).mean(), name='EMA30', line=dict(color='#f093fb',width=1)), row=1, col=1)
        
        if show_indicators and len(df)>20:
            u,m,l = calculate_bollinger(df['close'])
            fig.add_trace(go.Scatter(x=df.index,y=u,line=dict(color='rgba(255,255,255,0.2)',width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index,y=l,line=dict(color='rgba(255,255,255,0.2)',width=1),fill='tonexty',fillcolor='rgba(102,126,234,0.05)'), row=1, col=1)
        
        colors = ['#f5576c' if df['close'].iloc[i]<df['open'].iloc[i] else '#43e97b' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=colors, showlegend=False), row=2, col=1)
        
        if show_indicators:
            fig.add_trace(go.Scatter(x=df.index, y=calculate_rsi(df['close']), name='RSI', line=dict(color='#667eea',width=2)), row=3, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
        
        fig.update_layout(height=500 if show_indicators else 350, template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_ind:
        st.markdown("### 📊 Indicators")
        st.markdown("**RSI**"); st.progress(min(max(int(ind['rsi']),0),100)); st.caption(f"{ind['rsi']:.1f}")
        st.divider()
        st.markdown("**MACD**"); st.write(f"{'🟢 Bullish' if ind['macd_cross']>0 else '🔴 Bearish'}")
        st.divider()
        st.markdown("**Trend**"); st.write(f"{'📈' if ind['ema_trend']=='UP' else '📉'} {ind['ema_trend']}")
        st.divider()
        st.markdown("### 🎯 Signal")
        bg = {"LONG":"rgba(67,233,123,0.2)","SHORT":"rgba(245,87,108,0.2)","HOLD":"rgba(102,126,234,0.2)"}
        ic = {"LONG":"🟢","SHORT":"🔴","HOLD":"⚪"}
        st.markdown(f'<div class="signal-box" style="background:{bg[signal]}"><div style="font-size:28px">{ic[signal]}</div><div style="font-size:22px;font-weight:bold;color:#fff">{signal}</div><div style="font-size:13px;color:#888">{conf*100:.0f}%</div></div>', unsafe_allow_html=True)
else:
    st.warning("⏳ Loading chart...")
    if st.button("🔄 Retry"): st.cache_data.clear(); st.rerun()

st.divider()
st.markdown("⚠️ Educational purposes only. Not financial advice. | 📊 Data: Bybit API")
st.sidebar.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

if auto_refresh: time.sleep(10); st.rerun()