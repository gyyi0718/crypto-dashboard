# N-BEATS Paper Trading System

24시간 자동 실행되는 페이퍼 트레이딩 시스템입니다.

## 📁 파일 구조

```
paper_trading_daemon.py   # 백그라운드 트레이딩 데몬 (24시간 실행)
paper_trading_viewer.py   # 웹 대시보드 (결과 확인용)
start_daemon.bat          # 데몬 시작 스크립트 (Windows)
paper_trading.db          # SQLite DB (자동 생성)
logs/                     # 로그 폴더
```

## 🚀 실행 방법

### 1. 데몬 시작 (24시간 실행)

**방법 A: 배치 파일 실행**
```
start_daemon.bat
```

**방법 B: 직접 실행 (포그라운드)**
```bash
conda activate torch311
python paper_trading_daemon.py
```

**방법 C: 백그라운드 실행 (CMD 닫아도 유지)**
```bash
pythonw paper_trading_daemon.py
```

### 2. 웹 대시보드 실행 (결과 확인)

```bash
python -m streamlit run paper_trading_viewer.py
```

브라우저에서 `http://localhost:8501` 접속

## 🔧 필요한 파일

각 심볼별 모델 파일이 필요합니다:
- `nbeats_futures_model_BTCUSDT.pth` + `nbeats_scaler_BTCUSDT.json`
- `nbeats_futures_model_ETHUSDT.pth` + `nbeats_scaler_ETHUSDT.json`
- ... (다른 심볼들)

모델 파일이 없어도 해당 심볼은 트레이딩 비활성 상태로 실행됩니다.

## ⚙️ 설정 변경

`paper_trading_daemon.py` 상단의 설정값을 수정:

```python
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", ...]  # 트레이딩 심볼
INITIAL_BALANCE = 1000      # 초기 자본
LEVERAGE = 10               # 레버리지
POSITION_SIZE_PCT = 0.5     # 포지션 크기 (잔고의 50%)
STOP_LOSS_PCT = 0.02        # 손절 2%
TAKE_PROFIT_PCT = 0.03      # 익절 3%
CONF_THRESHOLD = 0.45       # 진입 신뢰도 임계값
INTERVAL_SEC = 60           # 체크 간격 (초)
```

## 📊 DB 구조

SQLite `paper_trading.db`:
- `accounts`: 계좌 잔고
- `positions`: 현재 포지션
- `trades`: 거래 기록
- `equity_history`: 자본 변화 히스토리
- `prediction_logs`: 예측 로그

## 🛑 데몬 종료

**방법 1: 작업 관리자**
- `pythonw.exe` 프로세스 종료

**방법 2: 명령어**
```bash
taskkill /F /IM pythonw.exe
```

## 📅 Windows 시작 시 자동 실행 (선택)

1. `Win + R` → `shell:startup` 입력
2. `start_daemon.bat` 바로가기 생성
3. 재부팅 시 자동 시작

## 📈 한 달 후 결과 확인

1. 웹 대시보드 실행
2. 히스토리 기간을 최대로 설정 (168시간 = 1주일)
3. 전체 요약에서 수익률 확인
4. Equity Curve에서 자본 변화 그래프 확인

더 긴 기간 보려면 `paper_trading_viewer.py`의 `view_hours` 최대값 수정
