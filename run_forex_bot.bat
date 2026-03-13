@echo off
setlocal

REM ---- FORCE VENV PYTHON ----
SET PYTHON=C:\Users\gisif\Desktop\Trader\.venv\Scripts\python.exe

REM ---- MOVE TO PROJECT ROOT ----
cd /d C:\Users\gisif\Desktop\Trader

REM ---- PROOF FILE ----
echo FOREX BAT RAN AT %DATE% %TIME% > FOREX_BAT_RAN.txt

REM ---- APPEND RUN HEADER + OUTPUT ----
echo.>> forex_output.log
echo ==============================>> forex_output.log
echo FOREX RUN START %DATE% %TIME%>> forex_output.log
echo ==============================>> forex_output.log
"%PYTHON%" forex_paper_bot.py >> forex_output.log 2>&1
set "BOT_RC=%ERRORLEVEL%"
if "%BOT_RC%"=="0" (
  echo FOREX RUN END %DATE% %TIME% STATUS=OK>> forex_output.log
) else (
  echo FOREX RUN END %DATE% %TIME% STATUS=FAIL RC=%BOT_RC%>> forex_output.log
  "%PYTHON%" forex_failure_alert.py --rc %BOT_RC% >> forex_output.log 2>&1
)
exit /b %BOT_RC%
