-- Supabase SQL Editor에서 실행하세요

-- 계좌 테이블
CREATE TABLE IF NOT EXISTS accounts (
  id SERIAL PRIMARY KEY,
  symbol TEXT UNIQUE NOT NULL,
  balance REAL DEFAULT 1000,
  initial_balance REAL DEFAULT 1000,
  created_at TIMESTAMP DEFAULT NOW()
);

-- 거래 기록 테이블
CREATE TABLE IF NOT EXISTS trades (
  id SERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  direction TEXT NOT NULL,
  entry_price REAL NOT NULL,
  exit_price REAL,
  qty REAL NOT NULL,
  pnl REAL DEFAULT 0,
  roe REAL DEFAULT 0,
  entry_time TIMESTAMP DEFAULT NOW(),
  exit_time TIMESTAMP,
  status TEXT DEFAULT 'open'
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);

-- 초기 계좌 데이터 (6개 코인)
INSERT INTO accounts (symbol, balance, initial_balance) VALUES 
  ('BTC-USD', 1000, 1000),
  ('ETH-USD', 1000, 1000),
  ('SOL-USD', 1000, 1000),
  ('XRP-USD', 1000, 1000),
  ('DOGE-USD', 1000, 1000),
  ('BNB-USD', 1000, 1000)
ON CONFLICT (symbol) DO NOTHING;

-- RLS (Row Level Security) 비활성화 (공개 앱용)
ALTER TABLE accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;

-- 모든 사용자 접근 허용 정책
CREATE POLICY "Allow all access to accounts" ON accounts FOR ALL USING (true);
CREATE POLICY "Allow all access to trades" ON trades FOR ALL USING (true);
