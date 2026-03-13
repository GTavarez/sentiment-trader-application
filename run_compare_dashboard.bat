@echo off

REM ---- FORCE VENV PYTHON ----
SET PYTHON=C:\Users\gisif\Desktop\Trader\.venv\Scripts\python.exe

REM ---- MOVE TO PROJECT ROOT ----
cd /d C:\Users\gisif\Desktop\Trader

REM ---- PROOF FILE ----
echo COMPARE DASH BAT RAN AT %DATE% %TIME% > COMPARE_DASH_BAT_RAN.txt

REM ---- START COMPARISON DASHBOARD ON PORT 8503 ----
start "" /b "%PYTHON%" -m streamlit run streamlit_compare_app.py --server.port 8503 --server.address 127.0.0.1 >> compare_dash_output.log 2>&1
exit /b 0
