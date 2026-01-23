# paper_trading_daemon.py
# -*- coding: utf-8 -*-
"""
N-BEATS Paper Trading Daemon
- 백그라운드에서 24시간 실행
- SQLite DB에 모든 거래/잔고 기록
"""

import os
import sys
import time
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn

# ==============================
# 로깅 설정
# ==============================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"daemon_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================
# 설정
# ==============================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT"]
TIMEFRAME = "1m"
LOOKBACK = 96
PRED_LEN = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 트레이딩 설정
INITIAL_BALANCE = 1000
LEVERAGE = 10
POSITION_SIZE_PCT = 0.5
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.03
MAX_HOLD_MINUTES = 60
CONF_THRESHOLD = 0.45

INTERVAL_SEC = 60  # 1분마다 체크

DB_PATH = "paper_trading.db"

# ==============================
# N-BEATS 모델 (동일)
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
# DB 관리
# ==============================

def init_db():
    """데이터베이스 초기화"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 계좌 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS accounts (
            symbol TEXT PRIMARY KEY,
            balance REAL,
            initial_balance REAL,
            created_at TEXT,
            updated_at TEXT
        )
    ''')
    
    # 포지션 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            qty REAL,
            margin REAL,
            sl REAL,
            tp REAL,
            entry_time TEXT,
            is_open INTEGER DEFAULT 1
        )
    ''')
    
    # 거래 기록 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            qty REAL,
            pnl REAL,
            roe REAL,
            entry_time TEXT,
            exit_time TEXT,
            reason TEXT
        )
    ''')
    
    # 자본 히스토리 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS equity_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timestamp TEXT,
            equity REAL,
            price REAL
        )
    ''')
    
    # 예측 로그 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timestamp TEXT,
            current_price REAL,
            predicted_price REAL,
            direction TEXT,
            confidence REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("DB 초기화 완료")


def get_or_create_account(symbol):
    """계좌 조회 또는 생성"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT balance FROM accounts WHERE symbol = ?", (symbol,))
    row = c.fetchone()
    
    if row is None:
        now = datetime.now().isoformat()
        c.execute(
            "INSERT INTO accounts (symbol, balance, initial_balance, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (symbol, INITIAL_BALANCE, INITIAL_BALANCE, now, now)
        )
        conn.commit()
        balance = INITIAL_BALANCE
        logger.info(f"{symbol} 계좌 생성: ${balance}")
    else:
        balance = row[0]
    
    conn.close()
    return balance


def update_balance(symbol, balance):
    """잔고 업데이트"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE accounts SET balance = ?, updated_at = ? WHERE symbol = ?",
        (balance, datetime.now().isoformat(), symbol)
    )
    conn.commit()
    conn.close()


def get_open_position(symbol):
    """열린 포지션 조회"""
    conn = sqlite3.connect(DB_PATH)
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
            'entry_time': datetime.fromisoformat(row[7])
        }
    return None


def open_position_db(symbol, direction, entry_price, qty, margin, sl, tp):
    """포지션 열기"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO positions (symbol, direction, entry_price, qty, margin, sl, tp, entry_time, is_open) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
        (symbol, direction, entry_price, qty, margin, sl, tp, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def close_position_db(symbol, position, exit_price, reason):
    """포지션 닫기"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # PnL 계산
    if position['direction'] == 'Long':
        pnl = (exit_price - position['entry_price']) * position['qty']
        roe = (exit_price / position['entry_price'] - 1) * LEVERAGE * 100
    else:
        pnl = (position['entry_price'] - exit_price) * position['qty']
        roe = (1 - exit_price / position['entry_price']) * LEVERAGE * 100
    
    # 포지션 닫기
    c.execute("UPDATE positions SET is_open = 0 WHERE id = ?", (position['id'],))
    
    # 거래 기록
    c.execute(
        "INSERT INTO trades (symbol, direction, entry_price, exit_price, qty, pnl, roe, entry_time, exit_time, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (symbol, position['direction'], position['entry_price'], exit_price, position['qty'], pnl, roe, position['entry_time'].isoformat(), datetime.now().isoformat(), reason)
    )
    
    conn.commit()
    conn.close()
    
    return pnl, roe


def save_equity(symbol, equity, price):
    """자본 히스토리 저장"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO equity_history (symbol, timestamp, equity, price) VALUES (?, ?, ?, ?)",
        (symbol, datetime.now().isoformat(), equity, price)
    )
    conn.commit()
    conn.close()


def save_prediction_log(symbol, current_price, predicted_price, direction, confidence):
    """예측 로그 저장"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO prediction_logs (symbol, timestamp, current_price, predicted_price, direction, confidence) VALUES (?, ?, ?, ?, ?, ?)",
        (symbol, datetime.now().isoformat(), current_price, predicted_price, direction, confidence)
    )
    conn.commit()
    conn.close()


# ==============================
# 데이터 가져오기
# ==============================

def fetch_futures_klines(symbol, interval="1m", limit=200):
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
        logger.error(f"{symbol} 데이터 로드 실패: {e}")
        return None


# ==============================
# 모델 로드 & 예측
# ==============================

def load_model(symbol):
    save_model = f"nbeats_futures_model_{symbol}.pth"
    scaler_path = f"nbeats_scaler_{symbol}.json"
    
    if not os.path.exists(scaler_path) or not os.path.exists(save_model):
        return None, None

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
    return model, scaler_info


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

    return direction, conf, pred_price


# ==============================
# 트레이딩 로직
# ==============================

def process_symbol(symbol, model, scaler_info):
    """심볼별 트레이딩 로직 처리"""
    
    # 데이터 로드
    df = fetch_futures_klines(symbol, TIMEFRAME, limit=LOOKBACK + 50)
    if df is None or len(df) < LOOKBACK:
        logger.warning(f"{symbol} 데이터 부족")
        return
    
    current_price = df["close"].iloc[-1]
    balance = get_or_create_account(symbol)
    position = get_open_position(symbol)
    
    # 예측
    if model and scaler_info:
        direction, conf, pred_price = predict(model, scaler_info, df)
        save_prediction_log(symbol, current_price, pred_price, direction, conf)
    else:
        direction, conf, pred_price = "Hold", 0, current_price
    
    # 현재 자본 계산
    current_equity = balance
    if position:
        if position['direction'] == 'Long':
            pnl = (current_price - position['entry_price']) * position['qty']
        else:
            pnl = (position['entry_price'] - current_price) * position['qty']
        current_equity += position['margin'] + pnl
    
    # 자본 히스토리 저장
    save_equity(symbol, current_equity, current_price)
    
    # 포지션 관리
    if position:
        held_min = (datetime.now() - position['entry_time']).total_seconds() / 60
        
        # SL/TP 체크
        close_reason = None
        if position['direction'] == 'Long':
            if current_price <= position['sl']:
                close_reason = "SL"
            elif current_price >= position['tp']:
                close_reason = "TP"
        else:
            if current_price >= position['sl']:
                close_reason = "SL"
            elif current_price <= position['tp']:
                close_reason = "TP"
        
        # Max Hold 체크
        if held_min >= MAX_HOLD_MINUTES:
            close_reason = "MaxHold"
        
        # 반대 신호
        if direction in ("Long", "Short") and direction != position['direction'] and conf >= CONF_THRESHOLD:
            close_reason = "Reverse"
        
        if close_reason:
            pnl, roe = close_position_db(symbol, position, current_price, close_reason)
            new_balance = balance + position['margin'] + pnl
            update_balance(symbol, new_balance)
            logger.info(f"{symbol} 포지션 종료: {close_reason} | PnL: ${pnl:+.2f} ({roe:+.1f}%)")
            return
    
    # 신규 진입
    if position is None and direction in ("Long", "Short") and conf >= CONF_THRESHOLD:
        margin = balance * POSITION_SIZE_PCT
        qty = (margin * LEVERAGE) / current_price
        
        if direction == "Long":
            sl = current_price * (1 - STOP_LOSS_PCT)
            tp = current_price * (1 + TAKE_PROFIT_PCT)
        else:
            sl = current_price * (1 + STOP_LOSS_PCT)
            tp = current_price * (1 - TAKE_PROFIT_PCT)
        
        open_position_db(symbol, direction, current_price, qty, margin, sl, tp)
        update_balance(symbol, balance - margin)
        logger.info(f"{symbol} 포지션 진입: {direction} @ ${current_price:.4f} (conf: {conf:.1%})")


# ==============================
# 메인 루프
# ==============================

def main():
    logger.info("="*60)
    logger.info("N-BEATS Paper Trading Daemon 시작")
    logger.info(f"심볼: {SYMBOLS}")
    logger.info(f"디바이스: {DEVICE}")
    logger.info("="*60)
    
    # DB 초기화
    init_db()
    
    # 모델 로드
    models = {}
    for symbol in SYMBOLS:
        model, scaler = load_model(symbol)
        if model:
            models[symbol] = (model, scaler)
            logger.info(f"✅ {symbol} 모델 로드 완료")
        else:
            models[symbol] = (None, None)
            logger.warning(f"⚠️ {symbol} 모델 없음 - 트레이딩 비활성")
    
    loop = 0
    
    while True:
        try:
            loop += 1
            logger.info(f"\n--- Loop {loop} | {datetime.now()} ---")
            
            for symbol in SYMBOLS:
                model, scaler = models[symbol]
                process_symbol(symbol, model, scaler)
            
            time.sleep(INTERVAL_SEC)
            
        except KeyboardInterrupt:
            logger.info("\n종료 신호 수신")
            break
        except Exception as e:
            logger.error(f"에러: {e}", exc_info=True)
            time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
