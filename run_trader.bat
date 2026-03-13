@echo off

REM ---- FORCE VENV PYTHON ----
SET PYTHON=C:\Users\gisif\Desktop\Trader\.venv\Scripts\python.exe

REM ---- MOVE TO PROJECT ROOT ----
cd /d C:\Users\gisif\Desktop\Trader

REM ---- PROOF FILE ----
echo BAT FILE RAN AT %DATE% %TIME% > BAT_RAN.txt

REM ---- APPEND RUN HEADER + OUTPUT ----
echo.>> bat_output.log
echo ==============================>> bat_output.log
echo RUN START %DATE% %TIME%>> bat_output.log
echo ==============================>> bat_output.log
"%PYTHON%" -m src.trader.main >> bat_output.log 2>&1
exit /b 0
