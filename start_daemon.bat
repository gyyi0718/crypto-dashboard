@echo off
REM ========================================
REM N-BEATS Paper Trading Daemon 실행
REM ========================================

REM conda 환경 활성화 (환경 이름에 맞게 수정)
call conda activate torch311

REM 백그라운드에서 실행 (pythonw 사용)
start /B pythonw paper_trading_daemon.py

echo ========================================
echo 데몬이 백그라운드에서 시작되었습니다.
echo 로그: logs/daemon_YYYYMMDD.log
echo 종료: 작업 관리자에서 pythonw 프로세스 종료
echo ========================================

pause
