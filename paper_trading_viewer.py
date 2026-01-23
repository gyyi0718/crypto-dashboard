# paper_trading_viewer.py
# -*- coding: utf-8 -*-
"""
N-BEATS Paper Trading Viewer
- 데몬이 저장한 DB를 읽어서 보여주기만 함
- 데몬과 별도로 실행
"""

import os
import sqlite3
from datetime import datetime, timedelta

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
    page_title="N-BEATS Paper Trading Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 설정
# ==============================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"]
DB_PATH = "paper_trading.db"
INITIAL_BALANCE = 1000

# ==============================
# DB 읽기 함수
# ==============================

def get_db_connection():
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH)


def get_account(symbol):
    conn = get_db_connection()
    if not conn:
        return {'balance': INITIAL_BALANCE, 'initial_balance': INITIAL_BALANCE}
    
    c = conn.cursor()
    c.execute("SELECT balance, initial_balance, created_at FROM accounts WHERE symbol = ?", (symbol,))
    row = c.fetchone()
    conn.close()
    
    if row:
        return {
            'balance': row[0],
            'initial_balance': row[1],
            'created_at': row[2]
        }
    return {'balance': INITIAL_BALANCE, 'initial_balance': INITIAL_BALANCE}


def get_open_position(symbol):
    conn = get_db_connection()
    if not conn:
        return None
    
    c = conn.cursor()
    c.execute(
        "SELECT id, direction, entry_price, qty, margin, sl, tp, entry_time FROM positions WHERE symbol = ? AND is_open = 1",
        (symbol,)
    )
    row = c.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row[0],
            'direction': row[1],
            'entry_price': row[2],
            'qty': row[3],
            'margin': row[4],
            'sl': row[5],
            'tp': row[6],
            'entry_time': row[7]
        }
    return None


def get_trades(symbol, limit=100):
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    df = pd.read_sql_query(
        f"SELECT * FROM trades WHERE symbol = ? ORDER BY exit_time DESC LIMIT ?",
        conn, params=(symbol, limit)
    )
    conn.close()
    return df


def get_all_trades():
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY exit_time DESC", conn)
    conn.close()
    return df


def get_equity_history(symbol, hours=24):
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    since = (datetime.now() - timedelta(hours=hours)).isoformat()
    df = pd.read_sql_query(
        "SELECT timestamp, equity, price FROM equity_history WHERE symbol = ? AND timestamp >= ? ORDER BY timestamp",
        conn, params=(symbol, since)
    )
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def get_prediction_logs(symbol, limit=100):
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    df = pd.read_sql_query(
        "SELECT * FROM prediction_logs WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
        conn, params=(symbol, limit)
    )
    conn.close()
    return df


def get_summary_all_symbols():
    """모든 심볼 요약"""
    summary = []
    
    for symbol in SYMBOLS:
        account = get_account(symbol)
        position = get_open_position(symbol)
        trades_df = get_trades(symbol)
        
        total_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0
        trade_count = len(trades_df)
        wins = len(trades_df[trades_df['pnl'] > 0]) if not trades_df.empty else 0
        win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
        
        # 현재 포지션 PnL
        pos_pnl = 0
        if position:
            price = get_current_price(symbol)
            if price:
                if position['direction'] == 'Long':
                    pos_pnl = (price - position['entry_price']) * position['qty']
                else:
                    pos_pnl = (position['entry_price'] - price) * position['qty']
        
        total_return = ((account['balance'] + pos_pnl) / account['initial_balance'] - 1) * 100
        
        summary.append({
            '심볼': symbol.replace("USDT", ""),
            '포지션': f"🟢 {position['direction']}" if position else "⚪ 없음",
            '잔고': f"${account['balance']:,.2f}",
            '미실현 PnL': f"${pos_pnl:+,.2f}" if position else "-",
            '실현 PnL': f"${total_pnl:+,.2f}",
            '수익률': f"{total_return:+.2f}%",
            '거래수': trade_count,
            '승률': f"{win_rate:.0f}%"
        })
    
    return pd.DataFrame(summary)


@st.cache_data(ttl=10)
def get_current_price(symbol):
    """현재 가격 조회"""
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
        r = requests.get(url, timeout=5)
        return float(r.json()['price'])
    except:
        return None


@st.cache_data(ttl=10)
def fetch_klines(symbol, limit=100):
    """캔들 데이터 조회"""
    try:
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": "1m", "limit": limit}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qv", "n", "tb", "tq", "ignore",
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = \
            df[["open", "high", "low", "close", "volume"]].astype(float)
        df = df.set_index("open_time")
        return df[["open", "high", "low", "close", "volume"]]
    except:
        return None


# ==============================
# 메인 UI
# ==============================

st.title("📊 N-BEATS Paper Trading Viewer")

# DB 상태 체크
if not os.path.exists(DB_PATH):
    st.error(f"❌ DB 파일 없음: {DB_PATH}")
    st.info("먼저 `python paper_trading_daemon.py`를 실행하세요.")
    st.stop()

st.success("✅ DB 연결됨")

# 사이드바
st.sidebar.title("⚙️ 설정")
selected_symbol = st.sidebar.selectbox("📌 심볼 선택", SYMBOLS, index=0)
view_hours = st.sidebar.slider("📅 히스토리 기간 (시간)", 1, 168, 24)
auto_refresh = st.sidebar.checkbox("🔄 자동 새로고침 (10초)", value=False)

if st.sidebar.button("🔄 새로고침"):
    st.cache_data.clear()
    st.rerun()

# ==============================
# 전체 요약
# ==============================

st.subheader("🌐 전체 심볼 요약")
summary_df = get_summary_all_symbols()
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.divider()

# ==============================
# 선택된 심볼 상세
# ==============================

st.subheader(f"📈 {selected_symbol} 상세")

# 데이터 로드
account = get_account(selected_symbol)
position = get_open_position(selected_symbol)
trades_df = get_trades(selected_symbol)
equity_df = get_equity_history(selected_symbol, hours=view_hours)
current_price = get_current_price(selected_symbol)
klines_df = fetch_klines(selected_symbol, limit=100)

# 상단 메트릭
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("💰 현재가", f"${current_price:,.4f}" if current_price else "N/A")

with col2:
    st.metric("💵 잔고", f"${account['balance']:,.2f}")

with col3:
    if position and current_price:
        if position['direction'] == 'Long':
            pos_pnl = (current_price - position['entry_price']) * position['qty']
        else:
            pos_pnl = (position['entry_price'] - current_price) * position['qty']
        st.metric("📊 미실현 PnL", f"${pos_pnl:+,.2f}")
    else:
        st.metric("📊 미실현 PnL", "-")

with col4:
    total_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0
    st.metric("💹 실현 PnL", f"${total_pnl:+,.2f}")

with col5:
    total_return = (account['balance'] / account['initial_balance'] - 1) * 100
    st.metric("📈 수익률", f"{total_return:+.2f}%")

st.divider()

# 차트 영역
col_chart, col_info = st.columns([2, 1])

with col_chart:
    # 가격 차트
    if klines_df is not None and not klines_df.empty:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price", "Volume")
        )

        # 캔들스틱
        fig.add_trace(
            go.Candlestick(
                x=klines_df.index,
                open=klines_df['open'],
                high=klines_df['high'],
                low=klines_df['low'],
                close=klines_df['close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # 포지션 진입가 표시
        if position:
            fig.add_hline(
                y=position['entry_price'], 
                line_dash="dash", 
                line_color="yellow",
                annotation_text=f"Entry: ${position['entry_price']:.4f}",
                row=1, col=1
            )
            fig.add_hline(
                y=position['sl'], 
                line_dash="dot", 
                line_color="red",
                annotation_text=f"SL: ${position['sl']:.4f}",
                row=1, col=1
            )
            fig.add_hline(
                y=position['tp'], 
                line_dash="dot", 
                line_color="green",
                annotation_text=f"TP: ${position['tp']:.4f}",
                row=1, col=1
            )

        # 거래량
        colors = ['red' if klines_df['close'].iloc[i] < klines_df['open'].iloc[i] else 'green' 
                  for i in range(len(klines_df))]
        fig.add_trace(
            go.Bar(x=klines_df.index, y=klines_df['volume'], marker_color=colors, showlegend=False),
            row=2, col=1
        )

        fig.update_layout(
            height=500,
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

with col_info:
    # 현재 포지션
    st.markdown("**📊 현재 포지션**")
    if position:
        pos_color = "🟢" if position['direction'] == 'Long' else "🔴"
        st.markdown(f"""
        {pos_color} **{position['direction']}**
        
        - 진입가: ${position['entry_price']:.4f}
        - SL: ${position['sl']:.4f}
        - TP: ${position['tp']:.4f}
        - 진입: {position['entry_time'][:19]}
        """)
    else:
        st.info("포지션 없음")
    
    st.divider()
    
    # 통계
    st.markdown("**📈 통계**")
    if not trades_df.empty:
        wins = len(trades_df[trades_df['pnl'] > 0])
        losses = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = wins / len(trades_df) * 100
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losses > 0 else 0
        
        st.metric("승률", f"{win_rate:.1f}%")
        st.metric("총 거래", f"{len(trades_df)}건")
        st.metric("평균 수익", f"${avg_win:+.2f}")
        st.metric("평균 손실", f"${avg_loss:+.2f}")
    else:
        st.info("거래 기록 없음")

st.divider()

# Equity Curve
st.subheader("💹 자본 변화 (Equity Curve)")

if not equity_df.empty:
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_df['timestamp'],
        y=equity_df['equity'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#667eea', width=2),
        fillcolor='rgba(102,126,234,0.3)',
        name='Equity'
    ))
    fig_equity.add_hline(
        y=account['initial_balance'], 
        line_dash="dash", 
        line_color="white",
        annotation_text=f"Initial: ${account['initial_balance']}"
    )
    fig_equity.update_layout(
        height=300,
        template='plotly_dark',
        margin=dict(l=0, r=0, t=0, b=0),
        yaxis_title="Equity ($)",
        xaxis_title="Time"
    )
    st.plotly_chart(fig_equity, use_container_width=True)
else:
    st.info("자본 히스토리가 아직 없습니다.")

st.divider()

# 거래 기록
st.subheader("📜 거래 기록")

if not trades_df.empty:
    display_df = trades_df[['direction', 'entry_price', 'exit_price', 'pnl', 'roe', 'reason', 'exit_time']].copy()
    display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:+.2f}")
    display_df['roe'] = display_df['roe'].apply(lambda x: f"{x:+.1f}%")
    display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.4f}")
    display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:.4f}")
    display_df.columns = ['방향', '진입가', '청산가', 'PnL', 'ROE', '사유', '청산시간']
    
    st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)
else:
    st.info("거래 기록이 없습니다.")

# 자동 새로고침
if auto_refresh:
    import time
    time.sleep(10)
    st.rerun()

# 푸터
st.sidebar.divider()
st.sidebar.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
