@echo off

REM ---- FORCE VENV PYTHON ----
SET PYTHON=C:\Users\gisif\Desktop\Trader\.venv\Scripts\python.exe

REM ---- MOVE TO PROJECT ROOT ----
cd /d C:\Users\gisif\Desktop\Trader

REM ---- PROOF FILE ----
echo FOREX DASH BAT RAN AT %DATE% %TIME% > FOREX_DASH_BAT_RAN.txt

REM ---- START FOREX DASHBOARD ON PORT 8502 ----
start "" /b "%PYTHON%" -m streamlit run streamlit_forex_app.py --server.port 8502 --server.address 127.0.0.1 >> forex_dash_output.log 2>&1
exit /b 0
