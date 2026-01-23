# streamlit_app.py
# -*- coding: utf-8 -*-
"""
Crypto Trading Dashboard with Supabase
- Yahoo Finance: 실시간 가격
- Supabase: 거래 기록 영구 저장
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from supabase import create_client, Client

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
    .trade-btn {
        width: 100%;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# Supabase 연결
# ==============================
@st.cache_resource
def init_supabase():
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_KEY")
    if url and key:
        return create_client(url, key)
    return None

supabase = init_supabase()

# ==============================
# 설정
# ==============================
COINS = {
    "BTC-USD": {"name": "Bitcoin", "icon": "₿"},
    "ETH-USD": {"name": "Ethereum", "icon": "Ξ"},
    "SOL-USD": {"name": "Solana", "icon": "◎"},
    "XRP-USD": {"name": "XRP", "icon": "✕"},
    "DOGE-USD": {"name": "Dogecoin", "icon": "Ð"},
    "BNB-USD": {"name": "BNB", "icon": "🔶"},
}

COIN_LIST = list(COINS.keys())
INITIAL_BALANCE = 1000
LEVERAGE = 10

# ==============================
# Supabase 함수
# ==============================

def get_account(symbol):
    """계좌 조회"""
    if not supabase:
        return {"balance": INITIAL_BALANCE, "initial_balance": INITIAL_BALANCE}
    try:
        result = supabase.table("accounts").select("*").eq("symbol", symbol).execute()
        if result.data:
            return result.data[0]
        # 없으면 생성
        supabase.table("accounts").insert({"symbol": symbol, "balance": INITIAL_BALANCE, "initial_balance": INITIAL_BALANCE}).execute()
        return {"balance": INITIAL_BALANCE, "initial_balance": INITIAL_BALANCE}
    except:
        return {"balance": INITIAL_BALANCE, "initial_balance": INITIAL_BALANCE}


def update_balance(symbol, balance):
    """잔고 업데이트"""
    if not supabase:
        return
    try:
        supabase.table("accounts").update({"balance": balance}).eq("symbol", symbol).execute()
    except:
        pass


def get_open_position(symbol):
    """열린 포지션 조회"""
    if not supabase:
        return None
    try:
        result = supabase.table("trades").select("*").eq("symbol", symbol).eq("status", "open").execute()
        return result.data[0] if result.data else None
    except:
        return None


def open_position(symbol, direction, price, qty, margin):
    """포지션 열기"""
    if not supabase:
        return
    try:
        supabase.table("trades").insert({
            "symbol": symbol,
            "direction": direction,
            "entry_price": price,
            "qty": qty,
            "pnl": 0,
            "roe": 0,
            "status": "open"
        }).execute()
    except Exception as e:
        st.error(f"포지션 열기 실패: {e}")


def close_position(symbol, position, exit_price):
    """포지션 닫기"""
    if not supabase or not position:
        return 0, 0
    try:
        entry_price = position['entry_price']
        qty = position['qty']
        direction = position['direction']
        
        if direction == 'Long':
            pnl = (exit_price - entry_price) * qty
            roe = (exit_price / entry_price - 1) * LEVERAGE * 100
        else:
            pnl = (entry_price - exit_price) * qty
            roe = (1 - exit_price / entry_price) * LEVERAGE * 100
        
        supabase.table("trades").update({
            "exit_price": exit_price,
            "pnl": pnl,
            "roe": roe,
            "exit_time": datetime.now().isoformat(),
            "status": "closed"
        }).eq("id", position['id']).execute()
        
        return pnl, roe
    except:
        return 0, 0


def get_trade_history(symbol, limit=20):
    """거래 내역 조회"""
    if not supabase:
        return []
    try:
        result = supabase.table("trades").select("*").eq("symbol", symbol).eq("status", "closed").order("exit_time", desc=True).limit(limit).execute()
        return result.data or []
    except:
        return []


def get_all_stats():
    """전체 통계"""
    if not supabase:
        return {}
    try:
        stats = {}
        for symbol in COIN_LIST:
            account = get_account(symbol)
            position = get_open_position(symbol)
            trades = supabase.table("trades").select("pnl").eq("symbol", symbol).eq("status", "closed").execute().data or []
            
            total_pnl = sum(t['pnl'] for t in trades)
            win_trades = len([t for t in trades if t['pnl'] > 0])
            total_trades = len(trades)
            win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
            
            stats[symbol] = {
                'balance': account['balance'],
                'initial': account.get('initial_balance', INITIAL_BALANCE),
                'total_pnl': total_pnl,
                'trades': total_trades,
                'win_rate': win_rate,
                'has_position': position is not None,
                'position': position
            }
        return stats
    except:
        return {}


# ==============================
# Yahoo Finance API
# ==============================

@st.cache_data(ttl=30)
def get_all_prices():
    """가격 조회"""
    try:
        tickers = yf.Tickers(" ".join(COIN_LIST))
        prices = {}
        for symbol in COIN_LIST:
            try:
                ticker = tickers.tickers[symbol]
                hist = ticker.history(period="2d")
                if len(hist) >= 2:
                    prev_close = hist['Close'].iloc[-2]
                    current = hist['Close'].iloc[-1]
                    change = ((current / prev_close) - 1) * 100
                else:
                    current = hist['Close'].iloc[-1] if len(hist) > 0 else 0
                    change = 0
                prices[symbol] = {'price': current, 'change': change, 'volume': hist['Volume'].iloc[-1] if len(hist) > 0 else 0}
            except:
                continue
        return prices
    except:
        return {}


@st.cache_data(ttl=60)
def fetch_history(symbol, period="1d", interval="1m"):
    """히스토리 조회"""
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty:
            return None
        df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        return df[['open', 'high', 'low', 'close', 'volume']]
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
    return macd, macd.ewm(span=signal).mean()


def get_trading_signal(df):
    if df is None or len(df) < 30:
        return "HOLD", 0.5, {'rsi': 50, 'macd_cross': 0, 'ema_trend': 'N/A', 'score': 0}
    
    close = df['close']
    rsi = calculate_rsi(close).iloc[-1]
    macd, macd_signal = calculate_macd(close)
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

st.title("📈 Crypto Paper Trading")
if auto_trade:
    st.caption("🤖 **AUTO TRADE 모드** | 시그널에 따라 자동 거래 중...")
else:
    st.caption("실시간 분석 + 모의 거래 • BTC, ETH, SOL, XRP, DOGE, BNB")

# Supabase 연결 확인
if not supabase:
    st.warning("⚠️ Supabase 미연결 - 거래 기록이 저장되지 않습니다. Secrets에 SUPABASE_URL, SUPABASE_KEY를 추가하세요.")

# 사이드바
st.sidebar.title("⚙️ Settings")
selected_coin = st.sidebar.selectbox("📌 Coin", COIN_LIST,
    format_func=lambda x: f"{COINS[x]['icon']} {COINS[x]['name']}")

period_map = {"1일": ("1d", "1m"), "5일": ("5d", "5m"), "1개월": ("1mo", "1h")}
selected_period = st.sidebar.selectbox("⏱️ Period", list(period_map.keys()))
period, interval = period_map[selected_period]

st.sidebar.divider()
st.sidebar.subheader("🤖 Auto Trade")
auto_trade = st.sidebar.toggle("자동 거래 활성화", value=False)
if auto_trade:
    st.sidebar.success("✅ 자동 거래 ON")
    st.sidebar.caption("시그널에 따라 자동 진입/청산")
    auto_refresh = True
else:
    auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (60s)", value=False)

st.sidebar.divider()
if st.sidebar.button("🔄 새로고침"): st.cache_data.clear(); st.rerun()

# ==============================
# 전체 포트폴리오 요약
# ==============================

st.subheader("💼 Portfolio Summary")

with st.spinner("Loading..."):
    prices = get_all_prices()
    all_stats = get_all_stats()

if prices and all_stats:
    # 요약 테이블
    summary_data = []
    total_balance = 0
    total_pnl = 0
    
    for symbol in COIN_LIST:
        info = COINS[symbol]
        stat = all_stats.get(symbol, {})
        price = prices.get(symbol, {}).get('price', 0)
        
        balance = stat.get('balance', INITIAL_BALANCE)
        pnl = stat.get('total_pnl', 0)
        position = stat.get('position')
        
        # 미실현 손익
        unrealized = 0
        if position and price:
            if position['direction'] == 'Long':
                unrealized = (price - position['entry_price']) * position['qty']
            else:
                unrealized = (position['entry_price'] - price) * position['qty']
        
        equity = balance + unrealized
        roi = (equity / INITIAL_BALANCE - 1) * 100
        
        total_balance += equity
        total_pnl += pnl + unrealized
        
        summary_data.append({
            '코인': f"{info['icon']} {info['name']}",
            '포지션': f"{'🟢 Long' if position and position['direction']=='Long' else '🔴 Short' if position else '⚪ -'}",
            '잔고': f"${balance:,.2f}",
            '미실현': f"${unrealized:+,.2f}" if position else "-",
            '실현 PnL': f"${pnl:+,.2f}",
            'ROI': f"{roi:+.1f}%",
            '거래': stat.get('trades', 0),
            '승률': f"{stat.get('win_rate', 0):.0f}%"
        })
    
    # 전체 통계
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 총 자본", f"${total_balance:,.2f}")
    col2.metric("📈 총 PnL", f"${total_pnl:+,.2f}")
    col3.metric("💵 초기자본", f"${INITIAL_BALANCE * len(COIN_LIST):,}")
    total_roi = (total_balance / (INITIAL_BALANCE * len(COIN_LIST)) - 1) * 100
    col4.metric("📊 총 ROI", f"{total_roi:+.1f}%")
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

st.divider()

# ==============================
# 선택 코인 상세
# ==============================

info = COINS[selected_coin]
st.subheader(f"{info['icon']} {info['name']} Trading")

df = fetch_history(selected_coin, period=period, interval=interval)
price = prices.get(selected_coin, {}).get('price', 0) if prices else 0
account = get_account(selected_coin)
position = get_open_position(selected_coin)

if df is not None and not df.empty:
    signal, conf, ind = get_trading_signal(df)
    
    # ========== AUTO TRADE 로직 ==========
    if auto_trade and supabase:
        trade_executed = False
        
        # 포지션 없을 때 - 시그널대로 진입
        if not position and signal in ["LONG", "SHORT"] and conf >= 0.6:
            margin = account['balance'] * 0.5  # 50% 사용
            qty = (margin * LEVERAGE) / price
            direction = "Long" if signal == "LONG" else "Short"
            
            open_position(selected_coin, direction, price, qty, margin)
            update_balance(selected_coin, account['balance'] - margin)
            st.toast(f"🤖 Auto: {direction} 진입 @ ${price:,.2f}")
            trade_executed = True
        
        # 포지션 있을 때 - 반대 시그널이면 청산 후 새 포지션
        elif position:
            should_close = False
            new_direction = None
            
            if position['direction'] == 'Long' and signal == "SHORT" and conf >= 0.6:
                should_close = True
                new_direction = "Short"
            elif position['direction'] == 'Short' and signal == "LONG" and conf >= 0.6:
                should_close = True
                new_direction = "Long"
            
            if should_close:
                # 기존 포지션 청산
                pnl, roe = close_position(selected_coin, position, price)
                margin_returned = position['entry_price'] * position['qty'] / LEVERAGE
                new_balance = account['balance'] + margin_returned + pnl
                update_balance(selected_coin, new_balance)
                st.toast(f"🤖 Auto: 청산 PnL ${pnl:+,.2f}")
                
                # 새 포지션 진입
                time.sleep(0.5)
                account = get_account(selected_coin)  # 잔고 다시 조회
                margin = account['balance'] * 0.5
                qty = (margin * LEVERAGE) / price
                open_position(selected_coin, new_direction, price, qty, margin)
                update_balance(selected_coin, account['balance'] - margin)
                st.toast(f"🤖 Auto: {new_direction} 진입 @ ${price:,.2f}")
                trade_executed = True
        
        if trade_executed:
            st.cache_data.clear()
            time.sleep(1)
            st.rerun()
    # ========== AUTO TRADE 끝 ==========
    
    # 상단 메트릭
    c1, c2, c3, c4, c5 = st.columns(5)
    chg = prices.get(selected_coin, {}).get('change', 0) if prices else 0
    c1.metric("💰 현재가", f"${price:,.2f}", f"{chg:+.2f}%")
    c2.metric("💵 잔고", f"${account['balance']:,.2f}")
    
    if position:
        if position['direction'] == 'Long':
            unrealized = (price - position['entry_price']) * position['qty']
        else:
            unrealized = (position['entry_price'] - price) * position['qty']
        c3.metric("📊 미실현", f"${unrealized:+,.2f}")
        c4.metric("📍 포지션", f"{position['direction']} @ ${position['entry_price']:,.2f}")
    else:
        c3.metric("📊 미실현", "-")
        c4.metric("📍 포지션", "없음")
    
    c5.metric(f"{'🟢' if signal=='LONG' else '🔴' if signal=='SHORT' else '⚪'} 시그널", signal)
    
    st.divider()
    
    # 차트 & 거래
    col_chart, col_trade = st.columns([2, 1])
    
    with col_chart:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            increasing_line_color='#43e97b', decreasing_line_color='#f5576c'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['close'].ewm(span=10).mean(), name='EMA10', 
            line=dict(color='#4facfe',width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['close'].ewm(span=30).mean(), name='EMA30', 
            line=dict(color='#f093fb',width=1)), row=1, col=1)
        
        # 포지션 표시
        if position:
            fig.add_hline(y=position['entry_price'], line_dash="dash", line_color="yellow", 
                         annotation_text=f"Entry: ${position['entry_price']:.2f}", row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=calculate_rsi(df['close']), name='RSI', 
            line=dict(color='#667eea',width=2)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        
        fig.update_layout(height=450, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_trade:
        st.markdown("### 🎮 거래")
        
        if position:
            # 포지션 있음 - 청산만 가능
            st.info(f"**{position['direction']}** 포지션 보유 중")
            st.write(f"진입가: ${position['entry_price']:,.2f}")
            st.write(f"수량: {position['qty']:.6f}")
            
            if position['direction'] == 'Long':
                unrealized = (price - position['entry_price']) * position['qty']
                roe = (price / position['entry_price'] - 1) * LEVERAGE * 100
            else:
                unrealized = (position['entry_price'] - price) * position['qty']
                roe = (1 - price / position['entry_price']) * LEVERAGE * 100
            
            color = "green" if unrealized >= 0 else "red"
            st.markdown(f"**미실현 PnL:** <span style='color:{color}'>${unrealized:+,.2f} ({roe:+.1f}%)</span>", unsafe_allow_html=True)
            
            if st.button("🔴 포지션 청산", use_container_width=True, type="primary"):
                pnl, roe = close_position(selected_coin, position, price)
                new_balance = account['balance'] + position['entry_price'] * position['qty'] / LEVERAGE + pnl
                update_balance(selected_coin, new_balance)
                st.success(f"청산 완료! PnL: ${pnl:+,.2f} ({roe:+.1f}%)")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()
        else:
            # 포지션 없음 - 진입 가능
            st.write(f"**사용 가능:** ${account['balance']:,.2f}")
            
            pct = st.slider("포지션 크기 (%)", 10, 100, 50, 10)
            margin = account['balance'] * pct / 100
            qty = (margin * LEVERAGE) / price
            
            st.write(f"마진: ${margin:,.2f}")
            st.write(f"수량: {qty:.6f}")
            st.write(f"레버리지: {LEVERAGE}x")
            
            col_long, col_short = st.columns(2)
            
            with col_long:
                if st.button("🟢 LONG", use_container_width=True):
                    open_position(selected_coin, "Long", price, qty, margin)
                    update_balance(selected_coin, account['balance'] - margin)
                    st.success(f"Long 진입 @ ${price:,.2f}")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
            
            with col_short:
                if st.button("🔴 SHORT", use_container_width=True):
                    open_position(selected_coin, "Short", price, qty, margin)
                    update_balance(selected_coin, account['balance'] - margin)
                    st.success(f"Short 진입 @ ${price:,.2f}")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
        
        st.divider()
        
        # 시그널
        st.markdown("### 🎯 AI Signal")
        bg = {"LONG":"rgba(67,233,123,0.2)","SHORT":"rgba(245,87,108,0.2)","HOLD":"rgba(102,126,234,0.2)"}
        ic = {"LONG":"🟢","SHORT":"🔴","HOLD":"⚪"}
        st.markdown(f'''
        <div class="signal-box" style="background:{bg[signal]}">
            <div style="font-size:24px">{ic[signal]}</div>
            <div style="font-size:18px;font-weight:bold;color:#fff">{signal}</div>
            <div style="font-size:12px;color:#888">{conf*100:.0f}%</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.caption(f"RSI: {ind['rsi']:.1f} | Trend: {ind['ema_trend']}")

    # 거래 내역
    st.divider()
    st.subheader("📜 거래 내역")
    
    trades = get_trade_history(selected_coin)
    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df = trade_df[['direction', 'entry_price', 'exit_price', 'pnl', 'roe', 'exit_time']]
        trade_df['pnl'] = trade_df['pnl'].apply(lambda x: f"${x:+,.2f}")
        trade_df['roe'] = trade_df['roe'].apply(lambda x: f"{x:+.1f}%")
        trade_df['entry_price'] = trade_df['entry_price'].apply(lambda x: f"${x:,.2f}")
        trade_df['exit_price'] = trade_df['exit_price'].apply(lambda x: f"${x:,.2f}" if x else "-")
        trade_df.columns = ['방향', '진입가', '청산가', 'PnL', 'ROE', '시간']
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
    else:
        st.info("거래 내역이 없습니다.")

else:
    st.warning("차트 데이터 로딩 중...")

st.divider()
st.caption("⚠️ 모의 거래입니다. 실제 자금이 아닙니다. | Data: Yahoo Finance")
st.sidebar.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

# Auto Refresh
if auto_trade:
    st.sidebar.warning("🤖 자동 거래 모드 (30초마다 체크)")
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()
elif auto_refresh:
    time.sleep(60)
    st.rerun()