# paper_trading_web.py
# -*- coding: utf-8 -*-
"""
N-BEATS Paper Trading - Streamlit Web Dashboard
"""

import os
import time
import json
import warnings
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ==============================
# 페이지 설정
# ==============================
st.set_page_config(
    page_title="N-BEATS Paper Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 설정
# ==============================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"]
TIMEFRAME = "1m"
LOOKBACK = 96
PRED_LEN = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 사이드바 설정
st.sidebar.title("⚙️ 설정")

# 심볼 선택
SYMBOL = st.sidebar.selectbox("📌 심볼 선택", SYMBOLS, index=0)

SAVE_MODEL = f"nbeats_futures_model_{SYMBOL}.pth"
SCALER_PATH = f"nbeats_scaler_{SYMBOL}.json"

INITIAL_BALANCE = st.sidebar.number_input("초기 자본 ($)", value=1000, min_value=100)
LEVERAGE = st.sidebar.slider("레버리지", 1, 20, 10)
POSITION_SIZE_PCT = st.sidebar.slider("포지션 크기 (%)", 10, 100, 50) / 100
STOP_LOSS_PCT = st.sidebar.slider("손절 (%)", 1, 10, 2) / 100
TAKE_PROFIT_PCT = st.sidebar.slider("익절 (%)", 1, 10, 3) / 100
CONF_THRESHOLD = st.sidebar.slider("신뢰도 임계값", 0.3, 0.7, 0.45)
INTERVAL_SEC = st.sidebar.slider("업데이트 간격 (초)", 5, 60, 10)

# ==============================
# N-BEATS 모델 구조
# ==============================

class GenericBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, theta_size: int):
        super().__init__()
        self.backcast_fc = nn.Linear(theta_size, backcast_size)
        self.forecast_fc = nn.Linear(theta_size, forecast_size)

    def backcast(self, theta):
        return self.backcast_fc(theta)

    def forecast(self, theta):
        return self.forecast_fc(theta)


class TrendBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, polynomial_degree: int = 3):
        super().__init__()
        self.polynomial_degree = polynomial_degree
        backcast_time = torch.arange(backcast_size).float() / backcast_size
        forecast_time = torch.arange(forecast_size).float() / forecast_size
        self.register_buffer('backcast_basis', self._make_polynomial_basis(backcast_time))
        self.register_buffer('forecast_basis', self._make_polynomial_basis(forecast_time))

    def _make_polynomial_basis(self, t):
        powers = torch.arange(self.polynomial_degree + 1).float()
        return t.unsqueeze(0) ** powers.unsqueeze(1)

    def backcast(self, theta):
        return torch.einsum('bp,pt->bt', theta, self.backcast_basis)

    def forecast(self, theta):
        return torch.einsum('bp,pt->bt', theta, self.forecast_basis)


class SeasonalityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, num_harmonics: int = 5):
        super().__init__()
        self.num_harmonics = num_harmonics
        backcast_time = torch.arange(backcast_size).float() / backcast_size
        forecast_time = torch.arange(forecast_size).float() / forecast_size
        self.register_buffer('backcast_basis', self._make_fourier_basis(backcast_time))
        self.register_buffer('forecast_basis', self._make_fourier_basis(forecast_time))

    def _make_fourier_basis(self, t):
        basis = []
        for k in range(1, self.num_harmonics + 1):
            basis.append(torch.sin(2 * np.pi * k * t))
            basis.append(torch.cos(2 * np.pi * k * t))
        return torch.stack(basis, dim=0)

    def backcast(self, theta):
        return torch.einsum('bp,pt->bt', theta, self.backcast_basis)

    def forecast(self, theta):
        return torch.einsum('bp,pt->bt', theta, self.forecast_basis)


class NBeatsBlock(nn.Module):
    def __init__(self, input_size: int, theta_size: int, basis_function: nn.Module,
                 num_layers: int = 4, hidden_size: int = 256):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.fc_stack = nn.Sequential(*layers)
        self.theta_f_fc = nn.Linear(hidden_size, theta_size)
        self.theta_b_fc = nn.Linear(hidden_size, theta_size)
        self.basis_function = basis_function

    def forward(self, x):
        h = self.fc_stack(x)
        theta_f = self.theta_f_fc(h)
        theta_b = self.theta_b_fc(h)
        backcast = self.basis_function.backcast(theta_b)
        forecast = self.basis_function.forecast(theta_f)
        return backcast, forecast


class NBeatsStack(nn.Module):
    def __init__(self, input_size: int, forecast_size: int, stack_type: str = "generic",
                 num_blocks: int = 3, num_layers: int = 4, hidden_size: int = 256,
                 polynomial_degree: int = 3, num_harmonics: int = 5):
        super().__init__()
        self.forecast_size = forecast_size

        if stack_type == "trend":
            theta_size = polynomial_degree + 1
            basis_fn = lambda: TrendBasis(input_size, forecast_size, polynomial_degree)
        elif stack_type == "seasonality":
            theta_size = 2 * num_harmonics
            basis_fn = lambda: SeasonalityBasis(input_size, forecast_size, num_harmonics)
        else:
            theta_size = hidden_size
            basis_fn = lambda: GenericBasis(input_size, forecast_size, theta_size)

        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, theta_size, basis_fn(), num_layers, hidden_size)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        residuals = x
        forecast = torch.zeros(x.shape[0], self.forecast_size, device=x.device)
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        return residuals, forecast


class NBeats(nn.Module):
    def __init__(self, enc_in: int = 4, seq_len: int = 96, pred_len: int = 60,
                 stack_types: list = ["trend", "seasonality", "generic"],
                 num_blocks_per_stack: int = 3, num_layers: int = 4,
                 hidden_size: int = 256, polynomial_degree: int = 3,
                 num_harmonics: int = 5, dropout: float = 0.1):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_size = seq_len * enc_in

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, seq_len),
        )

        self.stacks = nn.ModuleList([
            NBeatsStack(seq_len, pred_len, stack_type, num_blocks_per_stack,
                        num_layers, hidden_size, polynomial_degree, num_harmonics)
            for stack_type in stack_types
        ])

        self.price_head = nn.Sequential(
            nn.Linear(pred_len, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len),
        )

        self.dir_head = nn.Sequential(
            nn.Linear(pred_len, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len * 3),
        )

    def forward(self, x_enc, x_dec=None):
        B = x_enc.shape[0]
        x_flat = x_enc.reshape(B, -1)
        x_proj = self.input_proj(x_flat)

        residuals = x_proj
        forecast = torch.zeros(B, self.pred_len, device=x_enc.device)

        for stack in self.stacks:
            residuals, stack_forecast = stack(residuals)
            forecast = forecast + stack_forecast

        price_pred = self.price_head(forecast).unsqueeze(-1)
        dir_pred = self.dir_head(forecast).view(B, self.pred_len, 3)

        return price_pred, dir_pred


# ==============================
# 데이터 가져오기
# ==============================

@st.cache_data(ttl=5)
def fetch_futures_klines(symbol, interval="1m", limit=200):
    """Binance Futures에서 OHLCV 데이터 가져오기"""
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": min(limit, 1000)}
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qv", "n", "tb", "tq", "ignore",
        ])

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = \
            df[["open", "high", "low", "close", "volume"]].astype(float)

        df = df[["open_time", "open", "high", "low", "close", "volume"]]
        df = df.set_index("open_time")
        return df

    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None


# ==============================
# 기술적 지표 계산
# ==============================

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_ema(prices, period):
    if len(prices) < period:
        return np.mean(prices)
    multiplier = 2 / (period + 1)
    ema = np.mean(prices[:period])
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calculate_volume_ratio(volumes, period=20):
    if len(volumes) < period + 1:
        return 1.0
    current_vol = volumes[-1]
    avg_vol = np.mean(volumes[-(period + 1):-1])
    if avg_vol == 0:
        return 1.0
    return current_vol / avg_vol


# ==============================
# Session State 초기화 (심볼별)
# ==============================

def get_session_key(key):
    return f"{SYMBOL}_{key}"

if get_session_key('initialized') not in st.session_state:
    st.session_state[get_session_key('initialized')] = True
    st.session_state[get_session_key('balance')] = INITIAL_BALANCE
    st.session_state[get_session_key('position')] = None
    st.session_state[get_session_key('trades')] = []
    st.session_state[get_session_key('equity_history')] = [{'time': datetime.now(), 'equity': INITIAL_BALANCE}]

# 편의를 위한 래퍼
def get_state(key):
    return st.session_state.get(get_session_key(key))

def set_state(key, value):
    st.session_state[get_session_key(key)] = value


# ==============================
# 모델 로드
# ==============================

@st.cache_resource
def load_model(symbol):
    save_model = f"nbeats_futures_model_{symbol}.pth"
    scaler_path = f"nbeats_scaler_{symbol}.json"
    
    if not os.path.exists(scaler_path):
        return None, None, f"스케일러 파일 없음: {scaler_path}"
    if not os.path.exists(save_model):
        return None, None, f"모델 파일 없음: {save_model}"

    with open(scaler_path, "r") as f:
        scaler_info = json.load(f)

    checkpoint = torch.load(save_model, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_configs" in checkpoint:
        cfg = checkpoint["model_configs"]
        model = NBeats(
            enc_in=cfg.get("enc_in", 4),
            seq_len=cfg.get("seq_len", LOOKBACK),
            pred_len=cfg.get("pred_len", PRED_LEN),
            stack_types=cfg.get("stack_types", ["trend", "seasonality", "generic"]),
            num_blocks_per_stack=cfg.get("num_blocks_per_stack", 3),
            num_layers=cfg.get("num_layers", 4),
            hidden_size=cfg.get("hidden_size", 256),
        ).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = NBeats(enc_in=4, seq_len=LOOKBACK, pred_len=PRED_LEN).to(DEVICE)
        model.load_state_dict(checkpoint)

    model.eval()
    return model, scaler_info, None


def predict(model, scaler_info, df):
    closes = df["close"].values[-LOOKBACK:]
    vols = df["volume"].values[-LOOKBACK:]

    feature_mean = np.array(scaler_info["feature_mean"])
    feature_std = np.array(scaler_info["feature_std"])
    price_mean = scaler_info["price_mean"]
    price_std = scaler_info["price_std"]

    feat = np.column_stack([closes, vols, closes, vols])
    feat_norm = (feat - feature_mean) / feature_std

    x_enc = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        price_pred, dir_pred = model(x_enc)

    probs = torch.softmax(dir_pred[:, -1, :], dim=1).cpu().numpy()[0]
    best_cls = int(np.argmax(probs))

    if best_cls == 2:
        direction = "Long"
        conf = float(probs[2])
    elif best_cls == 1:
        direction = "Short"
        conf = float(probs[1])
    else:
        direction = "Hold"
        conf = float(probs[0])

    pred_price_norm = price_pred[0, -1, 0].item()
    pred_price = pred_price_norm * price_std + price_mean

    return direction, conf, pred_price, probs


# ==============================
# 메인 UI
# ==============================

st.title("🤖 N-BEATS Paper Trading Dashboard")
st.markdown(f"**{SYMBOL}** | Timeframe: {TIMEFRAME} | Device: {DEVICE}")

# 모델 로드 상태
model, scaler_info, error = load_model(SYMBOL)
if error:
    st.warning(f"⚠️ {error}")
    st.info(f"{SYMBOL} 모델 파일을 같은 디렉토리에 위치시켜주세요.")
    model_available = False
else:
    st.success(f"✅ {SYMBOL} 모델 로드 완료")
    model_available = True

# 데이터 로드
df = fetch_futures_klines(SYMBOL, TIMEFRAME, limit=LOOKBACK + 50)
if df is None or len(df) < LOOKBACK:
    st.error("데이터 로드 실패")
    st.stop()

current_price = df["close"].iloc[-1]
current_time = datetime.now()

# 예측 (모델 있을 때만)
if model_available:
    direction, conf, pred_price, probs = predict(model, scaler_info, df)
else:
    direction, conf, pred_price, probs = "N/A", 0, current_price, [0.33, 0.33, 0.34]

# 기술적 지표
rsi = calculate_rsi(df["close"].values)
ema_short = calculate_ema(df["close"].values, 10)
ema_long = calculate_ema(df["close"].values, 30)
vol_ratio = calculate_volume_ratio(df["volume"].values)

# ==============================
# 상단 메트릭
# ==============================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("💰 현재가", f"${current_price:,.4f}")

with col2:
    diff_pct = (pred_price - current_price) / current_price * 100
    st.metric("🔮 예측가", f"${pred_price:,.4f}", f"{diff_pct:+.2f}%")

with col3:
    dir_emoji = {"Long": "📈", "Short": "📉", "Hold": "➖", "N/A": "❓"}.get(direction, "➖")
    st.metric(f"{dir_emoji} 방향", direction, f"{conf*100:.1f}%")

with col4:
    st.metric("💵 잔고", f"${get_state('balance'):,.2f}")

with col5:
    total_pnl = sum(t["pnl"] for t in get_state('trades'))
    st.metric("📊 총 PnL", f"${total_pnl:+.2f}")

st.divider()

# ==============================
# 차트 섹션
# ==============================

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📈 가격 차트")
    
    # Candlestick 차트
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Price", "Volume", "RSI")
    )

    # 캔들스틱
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # EMA
    ema_short_series = df['close'].ewm(span=10).mean()
    ema_long_series = df['close'].ewm(span=30).mean()
    
    fig.add_trace(
        go.Scatter(x=df.index, y=ema_short_series, name='EMA 10', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=ema_long_series, name='EMA 30', line=dict(color='purple', width=1)),
        row=1, col=1
    )

    # 예측 가격 표시
    fig.add_hline(y=pred_price, line_dash="dash", line_color="green", 
                  annotation_text=f"Pred: ${pred_price:.2f}", row=1, col=1)

    # 거래량
    colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], marker_color=colors, name='Volume', showlegend=False),
        row=2, col=1
    )

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    
    fig.add_trace(
        go.Scatter(x=df.index, y=rsi_series, name='RSI', line=dict(color='blue', width=1)),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

    fig.update_layout(
        height=600,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("🔍 필터 상태")
    
    # RSI 게이지
    st.markdown("**RSI**")
    rsi_color = "🟢" if 30 < rsi < 70 else "🔴"
    st.progress(min(int(rsi), 100))
    st.caption(f"{rsi_color} {rsi:.1f}")

    # 거래량 비율
    st.markdown("**거래량 비율**")
    vol_bar = min(vol_ratio / 3, 1.0)
    vol_emoji = "🔥" if vol_ratio >= 2.0 else "✅" if vol_ratio >= 0.8 else "⚠️"
    st.progress(vol_bar)
    st.caption(f"{vol_emoji} {vol_ratio:.2f}x")

    # 추세
    st.markdown("**추세 (EMA)**")
    trend = "📈 상승" if ema_short > ema_long else "📉 하락"
    trend_strength = abs(ema_short - ema_long) / ema_long * 100
    st.write(f"{trend} ({trend_strength:.3f}%)")

    # 방향 확률
    st.markdown("**방향 확률**")
    prob_df = pd.DataFrame({
        '방향': ['Hold', 'Short', 'Long'],
        '확률': probs
    })
    
    fig_prob = go.Figure(go.Bar(
        x=prob_df['확률'],
        y=prob_df['방향'],
        orientation='h',
        marker_color=['gray', 'red', 'green']
    ))
    fig_prob.update_layout(
        height=150,
        template='plotly_dark',
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_range=[0, 1]
    )
    st.plotly_chart(fig_prob, use_container_width=True)

st.divider()

# ==============================
# 포지션 & 거래 기록
# ==============================

col_pos, col_trades = st.columns(2)

with col_pos:
    st.subheader("📊 현재 포지션")
    
    if get_state('position'):
        pos = get_state('position')
        pnl = (current_price - pos['entry_price']) * pos['qty'] if pos['direction'] == 'Long' else (pos['entry_price'] - current_price) * pos['qty']
        roe = pnl / pos['margin'] * 100
        
        pos_color = "🟢" if pnl >= 0 else "🔴"
        
        st.markdown(f"""
        **{pos['direction']}** @ ${pos['entry_price']:.4f}
        
        {pos_color} PnL: ${pnl:+.2f} ({roe:+.1f}%)
        
        - SL: ${pos['sl']:.4f}
        - TP: ${pos['tp']:.4f}
        - 진입: {pos['entry_time'].strftime('%H:%M:%S')}
        """)
        
        if st.button("🔒 포지션 종료", key=f"close_{SYMBOL}"):
            new_balance = get_state('balance') + pos['margin'] + pnl
            set_state('balance', new_balance)
            trades = get_state('trades')
            trades.append({
                'direction': pos['direction'],
                'entry_price': pos['entry_price'],
                'exit_price': current_price,
                'pnl': pnl,
                'roe': roe,
                'entry_time': pos['entry_time'],
                'exit_time': datetime.now(),
                'reason': 'Manual'
            })
            set_state('trades', trades)
            set_state('position', None)
            st.rerun()
    else:
        st.info("포지션 없음")
        
        col_long, col_short = st.columns(2)
        
        with col_long:
            if st.button("📈 LONG 진입", type="primary", key=f"long_{SYMBOL}"):
                margin = get_state('balance') * POSITION_SIZE_PCT
                qty = (margin * LEVERAGE) / current_price
                set_state('position', {
                    'direction': 'Long',
                    'entry_price': current_price,
                    'qty': qty,
                    'margin': margin,
                    'sl': current_price * (1 - STOP_LOSS_PCT),
                    'tp': current_price * (1 + TAKE_PROFIT_PCT),
                    'entry_time': datetime.now()
                })
                set_state('balance', get_state('balance') - margin)
                st.rerun()
        
        with col_short:
            if st.button("📉 SHORT 진입", type="secondary", key=f"short_{SYMBOL}"):
                margin = get_state('balance') * POSITION_SIZE_PCT
                qty = (margin * LEVERAGE) / current_price
                set_state('position', {
                    'direction': 'Short',
                    'entry_price': current_price,
                    'qty': qty,
                    'margin': margin,
                    'sl': current_price * (1 + STOP_LOSS_PCT),
                    'tp': current_price * (1 - TAKE_PROFIT_PCT),
                    'entry_time': datetime.now()
                })
                set_state('balance', get_state('balance') - margin)
                st.rerun()

with col_trades:
    st.subheader("📜 거래 기록")
    
    if get_state('trades'):
        trades_df = pd.DataFrame(get_state('trades'))
        trades_df['pnl_str'] = trades_df['pnl'].apply(lambda x: f"${x:+.2f}")
        trades_df['roe_str'] = trades_df['roe'].apply(lambda x: f"{x:+.1f}%")
        trades_df['entry_time_str'] = trades_df['entry_time'].apply(lambda x: x.strftime('%H:%M'))
        trades_df['exit_time_str'] = trades_df['exit_time'].apply(lambda x: x.strftime('%H:%M'))
        
        display_df = trades_df[['direction', 'entry_price', 'exit_price', 'pnl_str', 'roe_str', 'reason']].tail(10)
        display_df.columns = ['방향', '진입가', '청산가', 'PnL', 'ROE', '사유']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # 통계
        wins = sum(1 for t in get_state('trades') if t['pnl'] > 0)
        total = len(get_state('trades'))
        win_rate = wins / total * 100 if total > 0 else 0
        
        st.metric("승률", f"{win_rate:.1f}% ({wins}/{total})")
    else:
        st.info("거래 기록 없음")

st.divider()

# ==============================
# Equity Curve
# ==============================

st.subheader("💹 자본 변화 (Equity Curve)")

# 현재 자본 기록
current_equity = get_state('balance')
if get_state('position'):
    pos = get_state('position')
    pnl = (current_price - pos['entry_price']) * pos['qty'] if pos['direction'] == 'Long' else (pos['entry_price'] - current_price) * pos['qty']
    current_equity += pos['margin'] + pnl

equity_history = get_state('equity_history')
equity_history.append({'time': current_time, 'equity': current_equity})

# 최근 100개만 유지
if len(equity_history) > 100:
    equity_history = equity_history[-100:]
set_state('equity_history', equity_history)

equity_df = pd.DataFrame(equity_history)

fig_equity = go.Figure()
fig_equity.add_trace(go.Scatter(
    x=equity_df['time'],
    y=equity_df['equity'],
    mode='lines',
    fill='tozeroy',
    line=dict(color='#667eea', width=2),
    fillcolor='rgba(102,126,234,0.3)'
))
fig_equity.add_hline(y=INITIAL_BALANCE, line_dash="dash", line_color="white", 
                     annotation_text=f"Initial: ${INITIAL_BALANCE}")
fig_equity.update_layout(
    height=250,
    template='plotly_dark',
    margin=dict(l=0, r=0, t=0, b=0),
    yaxis_title="Equity ($)"
)
st.plotly_chart(fig_equity, use_container_width=True)

# ==============================
# 멀티 심볼 요약 (전체 현황)
# ==============================

st.divider()
st.subheader("🌐 전체 심볼 현황")

summary_data = []
for sym in SYMBOLS:
    sym_balance = st.session_state.get(f"{sym}_balance", INITIAL_BALANCE)
    sym_trades = st.session_state.get(f"{sym}_trades", [])
    sym_position = st.session_state.get(f"{sym}_position", None)
    
    total_pnl = sum(t["pnl"] for t in sym_trades) if sym_trades else 0
    trade_count = len(sym_trades)
    wins = sum(1 for t in sym_trades if t["pnl"] > 0) if sym_trades else 0
    win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
    
    pos_status = "🟢" if sym_position else "⚪"
    
    summary_data.append({
        '심볼': sym.replace("USDT", ""),
        '포지션': pos_status,
        '잔고': f"${sym_balance:,.2f}",
        '총 PnL': f"${total_pnl:+.2f}",
        '거래수': trade_count,
        '승률': f"{win_rate:.0f}%"
    })

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ==============================
# 자동 새로고침
# ==============================

st.sidebar.divider()
auto_refresh = st.sidebar.checkbox("🔄 자동 새로고침", value=False)
if auto_refresh:
    st.sidebar.info(f"{INTERVAL_SEC}초마다 새로고침")
    time.sleep(INTERVAL_SEC)
    st.rerun()

# 수동 새로고침 버튼
if st.sidebar.button("🔄 새로고침"):
    st.rerun()

# 리셋 버튼
if st.sidebar.button(f"🗑️ {SYMBOL} 초기화"):
    set_state('balance', INITIAL_BALANCE)
    set_state('position', None)
    set_state('trades', [])
    set_state('equity_history', [{'time': datetime.now(), 'equity': INITIAL_BALANCE}])
    st.rerun()

# 전체 리셋
if st.sidebar.button("🗑️ 전체 초기화"):
    for sym in SYMBOLS:
        st.session_state[f"{sym}_balance"] = INITIAL_BALANCE
        st.session_state[f"{sym}_position"] = None
        st.session_state[f"{sym}_trades"] = []
        st.session_state[f"{sym}_equity_history"] = [{'time': datetime.now(), 'equity': INITIAL_BALANCE}]
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(f"마지막 업데이트: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
