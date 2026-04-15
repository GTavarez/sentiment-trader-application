@echo off
setlocal

REM ---- PROJECT ROOT ----
set "PROJECT_ROOT=C:\Users\gisif\Desktop\Trader"
cd /d "%PROJECT_ROOT%"

REM ---- CONFIG ----
REM Set this in your user env to your desktop share, e.g. \\DESKTOP-NAME\TraderShare
if "%DESKTOP_TRADER_SHARE%"=="" set "DESKTOP_TRADER_SHARE=\\REPLACE_WITH_DESKTOP_NAME\TraderShare"

echo ==============================>> sync_output.log
echo SYNC START %DATE% %TIME%>> sync_output.log
echo SOURCE=%DESKTOP_TRADER_SHARE%>> sync_output.log

if not exist "%DESKTOP_TRADER_SHARE%\" (
  echo SYNC SKIP: source share not reachable>> sync_output.log
  echo SYNC END %DATE% %TIME% STATUS=SKIP>> sync_output.log
  exit /b 0
)

if not exist "%PROJECT_ROOT%\data" mkdir "%PROJECT_ROOT%\data"

robocopy "%DESKTOP_TRADER_SHARE%\data" "%PROJECT_ROOT%\data" trader.db forex_trader.db /R:1 /W:1 /NFL /NDL /NJH /NJS /NP >nul
set "RC_DATA=%ERRORLEVEL%"

robocopy "%DESKTOP_TRADER_SHARE%" "%PROJECT_ROOT%" bat_output.log forex_output.log /R:1 /W:1 /NFL /NDL /NJH /NJS /NP >nul
set "RC_ROOT=%ERRORLEVEL%"

if %RC_DATA% GEQ 8 (
  echo SYNC WARN: data copy returned rc=%RC_DATA%>> sync_output.log
)
if %RC_ROOT% GEQ 8 (
  echo SYNC WARN: root copy returned rc=%RC_ROOT%>> sync_output.log
)

echo SYNC END %DATE% %TIME% STATUS=OK>> sync_output.log
exit /b 0
