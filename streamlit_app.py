# streamlit_app.py
# -*- coding: utf-8 -*-
"""
Crypto Trading Dashboard
- Streamlit Cloud 배포용
- CoinGecko API 사용 (전세계 접속 가능)
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

# 다크 테마 CSS
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
COINS = {
    "bitcoin": {"symbol": "BTC", "name": "Bitcoin", "icon": "₿"},
    "ethereum": {"symbol": "ETH", "name": "Ethereum", "icon": "Ξ"},
    "solana": {"symbol": "SOL", "name": "Solana", "icon": "◎"},
    "ripple": {"symbol": "XRP", "name": "XRP", "icon": "✕"},
    "dogecoin": {"symbol": "DOGE", "name": "Dogecoin", "icon": "Ð"},
    "binancecoin": {"symbol": "BNB", "name": "BNB", "icon": "🔶"},
}

COIN_IDS = list(COINS.keys())

# ==============================
# API 함수
# ==============================

@st.cache_data(ttl=30)
def get_all_prices():
    """CoinGecko에서 모든 코인 가격 조회"""
    try:
        ids = ",".join(COIN_IDS)
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": ids,
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_24hr_vol": "true",
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        prices = {}
        for coin_id, info in COINS.items():
            if coin_id in data:
                coin_data = data[coin_id]
                prices[coin_id] = {
                    'price': coin_data.get('usd', 0),
                    'change': coin_data.get('usd_24h_change', 0),
                    'volume': coin_data.get('usd_24h_vol', 0),
                }
        return prices
    except Exception as e:
        st.error(f"가격 조회 실패: {e}")
        return {}


@st.cache_data(ttl=60)
def fetch_ohlc(coin_id, days=1):
    """CoinGecko에서 OHLC 데이터 조회"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        params = {"vs_currency": "usd", "days": days}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        if not data:
            return None
            
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        return df
    except Exception as e:
        return None


@st.cache_data(ttl=60)
def fetch_market_chart(coin_id, days=1):
    """CoinGecko에서 가격 히스토리 조회"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        
        if not prices:
            return None
        
        df = pd.DataFrame(prices, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        
        if volumes:
            vol_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
            vol_df["timestamp"] = pd.to_datetime(vol_df["timestamp"], unit="ms")
            vol_df = vol_df.set_index("timestamp")
            df = df.join(vol_df, how="left")
        
        return df
    except Exception as e:
        return None


# ==============================
# 기술적 지표
# ==============================

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal, macd - macd_signal


def calculate_bollinger(prices, period=20, std=2):
    sma = prices.rolling(window=period).mean()
    std_dev = prices.rolling(window=period).std()
    return sma + (std_dev * std), sma, sma - (std_dev * std)


def get_trading_signal(df):
    """트레이딩 시그널 생성"""
    if df is None or len(df) < 30:
        return "HOLD", 0.5, {'rsi': 50, 'macd_cross': 0, 'ema_trend': 'N/A', 'score': 0}
    
    close = df['close']
    rsi = calculate_rsi(close).iloc[-1]
    
    macd, macd_signal, _ = calculate_macd(close)
    macd_cross = macd.iloc[-1] - macd_signal.iloc[-1]
    
    if pd.isna(macd_cross):
        macd_cross = 0
    
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
    
    if score >= 2:
        signal, confidence = "LONG", min(0.5 + score * 0.1, 0.9)
    elif score <= -2:
        signal, confidence = "SHORT", min(0.5 + abs(score) * 0.1, 0.9)
    else:
        signal, confidence = "HOLD", 0.5
    
    return signal, confidence, {'rsi': rsi, 'macd_cross': macd_cross, 'ema_trend': ema_trend, 'score': score}


# ==============================
# 메인 UI
# ==============================

st.title("📈 Crypto Trading Dashboard")
st.caption("Real-time cryptocurrency analysis • BTC, ETH, SOL, XRP, DOGE, BNB")

# 사이드바
st.sidebar.title("⚙️ Settings")
selected_coin = st.sidebar.selectbox(
    "📌 Select Coin", COIN_IDS,
    format_func=lambda x: f"{COINS[x]['icon']} {COINS[x]['name']} ({COINS[x]['symbol']})"
)
chart_days = st.sidebar.selectbox("📅 Chart Period", [1, 7, 14, 30], index=0, format_func=lambda x: f"{x} day(s)")
show_indicators = st.sidebar.checkbox("📊 Show Indicators", value=True)
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=False)

if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.rerun()

# ==============================
# 전체 코인 현황
# ==============================

st.subheader("🌐 Market Overview")
prices = get_all_prices()

if prices:
    cols = st.columns(6)
    for i, coin_id in enumerate(COIN_IDS):
        if coin_id in prices:
            data = prices[coin_id]
            info = COINS[coin_id]
            with cols[i]:
                change = data['change'] or 0
                change_color = "profit" if change >= 0 else "loss"
                change_icon = "▲" if change >= 0 else "▼"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 20px;">{info['icon']}</div>
                    <div style="font-size: 11px; color: #888;">{info['name']}</div>
                    <div style="font-size: 16px; font-weight: bold; color: #fff;">${data['price']:,.4f}</div>
                    <div class="{change_color}">{change_icon} {change:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
else:
    st.warning("가격 데이터를 불러오는 중...")

st.divider()

# ==============================
# 선택 코인 상세
# ==============================

coin_info = COINS[selected_coin]
st.subheader(f"{coin_info['icon']} {coin_info['name']} ({coin_info['symbol']}) Analysis")

# 데이터 로드
df = fetch_market_chart(selected_coin, days=chart_days)
ohlc_df = fetch_ohlc(selected_coin, days=chart_days)
current_price = prices.get(selected_coin, {}).get('price', 0)

if df is not None and not df.empty:
    signal, confidence, indicators = get_trading_signal(df)
    
    # 메트릭
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        change = prices.get(selected_coin, {}).get('change', 0) or 0
        st.metric("💰 Price", f"${current_price:,.4f}", f"{change:+.2f}%")
    
    with col2:
        sig_icon = {"LONG": "🟢", "SHORT": "🔴", "HOLD": "⚪"}.get(signal, "⚪")
        st.metric(f"{sig_icon} Signal", signal, f"{confidence*100:.0f}%")
    
    with col3:
        rsi = indicators.get('rsi', 50)
        rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
        st.metric("📊 RSI", f"{rsi:.1f}", rsi_status)
    
    with col4:
        st.metric("📈 Trend", indicators.get('ema_trend', 'N/A'))
    
    with col5:
        vol = prices.get(selected_coin, {}).get('volume', 0) or 0
        st.metric("📊 24h Vol", f"${vol/1e9:.2f}B" if vol > 1e9 else f"${vol/1e6:.1f}M")
    
    st.divider()
    
    # 차트 & 지표
    col_chart, col_ind = st.columns([2, 1])
    
    with col_chart:
        if ohlc_df is not None and not ohlc_df.empty:
            rows = 3 if show_indicators else 2
            heights = [0.5, 0.25, 0.25] if show_indicators else [0.7, 0.3]
            titles = ("Price", "Volume", "RSI") if show_indicators else ("Price", "Volume")
            
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                               row_heights=heights, subplot_titles=titles)
            
            # 캔들스틱
            fig.add_trace(go.Candlestick(
                x=ohlc_df.index, open=ohlc_df['open'], high=ohlc_df['high'],
                low=ohlc_df['low'], close=ohlc_df['close'],
                name='OHLC', increasing_line_color='#43e97b', decreasing_line_color='#f5576c'
            ), row=1, col=1)
            
            # EMA
            close_prices = ohlc_df['close']
            fig.add_trace(go.Scatter(x=ohlc_df.index, y=close_prices.ewm(span=10).mean(),
                                    name='EMA 10', line=dict(color='#4facfe', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=ohlc_df.index, y=close_prices.ewm(span=30).mean(),
                                    name='EMA 30', line=dict(color='#f093fb', width=1)), row=1, col=1)
            
            if show_indicators and len(close_prices) > 20:
                upper, middle, lower = calculate_bollinger(close_prices)
                fig.add_trace(go.Scatter(x=ohlc_df.index, y=upper, name='BB Upper',
                                        line=dict(color='rgba(255,255,255,0.2)', width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=ohlc_df.index, y=lower, name='BB Lower',
                                        line=dict(color='rgba(255,255,255,0.2)', width=1),
                                        fill='tonexty', fillcolor='rgba(102,126,234,0.05)'), row=1, col=1)
            
            # 거래량
            if 'volume' in df.columns:
                fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color='#667eea',
                                    name='Volume', showlegend=False), row=2, col=1)
            
            if show_indicators:
                rsi_series = calculate_rsi(close_prices)
                fig.add_trace(go.Scatter(x=ohlc_df.index, y=rsi_series, name='RSI',
                                        line=dict(color='#667eea', width=2)), row=3, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
            
            fig.update_layout(
                height=550 if show_indicators else 400,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_rangeslider_visible=False,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # 라인 차트 (OHLC 없을 때)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines',
                                    name='Price', line=dict(color='#667eea', width=2),
                                    fill='tozeroy', fillcolor='rgba(102,126,234,0.1)'))
            fig.update_layout(height=400, template='plotly_dark',
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    with col_ind:
        st.markdown("### 📊 Indicators")
        
        # RSI
        st.markdown("**RSI (14)**")
        rsi_val = indicators.get('rsi', 50)
        st.progress(min(max(int(rsi_val), 0), 100))
        st.caption(f"{rsi_val:.1f} - {'Oversold 📉' if rsi_val < 30 else 'Overbought 📈' if rsi_val > 70 else 'Neutral'}")
        
        st.divider()
        
        # MACD
        st.markdown("**MACD**")
        macd_cross = indicators.get('macd_cross', 0)
        macd_status = "🟢 Bullish" if macd_cross > 0 else "🔴 Bearish"
        st.write(f"Status: {macd_status}")
        
        st.divider()
        
        # EMA Trend
        st.markdown("**EMA Trend**")
        ema_trend = indicators.get('ema_trend', 'N/A')
        trend_icon = "📈" if ema_trend == "UP" else "📉"
        st.write(f"{trend_icon} {ema_trend}")
        
        st.divider()
        
        # 시그널 박스
        st.markdown("### 🎯 Signal")
        sig_bg = {"LONG": "rgba(67,233,123,0.2)", "SHORT": "rgba(245,87,108,0.2)", "HOLD": "rgba(102,126,234,0.2)"}
        sig_icon = {"LONG": "🟢", "SHORT": "🔴", "HOLD": "⚪"}
        
        st.markdown(f"""
        <div class="signal-box" style="background: {sig_bg.get(signal, sig_bg['HOLD'])}">
            <div style="font-size: 28px;">{sig_icon.get(signal, '⚪')}</div>
            <div style="font-size: 22px; font-weight: bold; color: #fff;">{signal}</div>
            <div style="font-size: 13px; color: #888;">Confidence: {confidence*100:.0f}%</div>
            <div style="font-size: 11px; color: #666; margin-top: 8px;">Score: {indicators.get('score', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.warning("⏳ 차트 데이터를 불러오는 중... 잠시 후 새로고침 해주세요.")
    if st.button("🔄 다시 시도"):
        st.cache_data.clear()
        st.rerun()

# ==============================
# 하단
# ==============================

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### ℹ️ About")
    st.markdown("""
    Real-time crypto analysis using technical indicators:
    - **RSI**: Oversold < 30, Overbought > 70
    - **MACD**: Trend momentum  
    - **EMA**: 10/30 period crossover
    - **Bollinger Bands**: Volatility
    
    ⚠️ Educational purposes only. Not financial advice.
    """)

with col2:
    st.markdown("### 🔗 Links")
    st.markdown("""
    - 📂 [GitHub](https://github.com/gyyi0718/crypto-dashboard)
    - 📊 Data: [CoinGecko API](https://www.coingecko.com/)
    """)

st.sidebar.divider()
st.sidebar.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.caption("📊 Data: CoinGecko (Free API)")

if auto_refresh:
    time.sleep(30)
    st.rerun()