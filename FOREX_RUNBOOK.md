## Forex Paper Bot Runbook

This bot is separate from the equities trader and runs in paper mode only.

### Files
- `forex_paper_bot.py`: Forex paper strategy runner
- `run_forex_bot.bat`: Windows launcher with append-only logs
- `data/forex_trader.db`: Separate SQLite DB for forex trades/positions
- `forex_output.log`: Forex run logs

### Quick Start
Run once:
```powershell
python forex_paper_bot.py
```

Or with venv:
```powershell
.\.venv\Scripts\python.exe forex_paper_bot.py
```

Run via batch:
```powershell
.\run_forex_bot.bat
```

Create/update hourly bot task:
```powershell
schtasks /Create /TN ForexTrader /SC HOURLY /MO 1 /ST 13:05 /TR "C:\Windows\System32\cmd.exe /c C:\Users\gisif\Desktop\Trader\run_forex_bot.bat" /RU gisif /F
```

Run once on demand:
```powershell
schtasks /Run /TN ForexTrader
```

### Visual Dashboard
Run forex dashboard:
```powershell
streamlit run streamlit_forex_app.py --server.port 8502
```
Open:
`http://127.0.0.1:8502`

Auto-launch batch:
```powershell
.\run_forex_dashboard.bat
```

Create startup task (runs at logon):
```powershell
schtasks /Create /TN ForexDashboard /SC ONLOGON /TR "C:\Windows\System32\cmd.exe /c C:\Users\gisif\Desktop\Trader\run_forex_dashboard.bat" /RU gisif /F
```

### Environment Variables
All values are optional.

```env
FOREX_MODE=paper
FOREX_PAIRS=USD/CAD,EUR/USD,GBP/USD,USD/JPY
FOREX_SIGNAL_THRESHOLD_PCT=0.002
FOREX_NOTIONAL_USD=1000
FOREX_MAX_HOLD_DAYS=3
FOREX_READ_ONLY=false
FOREX_API_BASE=https://api.frankfurter.app
FOREX_DB_PATH=data/forex_trader.db
FOREX_HTTP_TIMEOUT_S=15
FOREX_HTTP_RETRIES=3
FOREX_HTTP_BACKOFF_S=1.5
```

### Strategy (Current)
- Pulls latest and previous-business-day FX rates from Frankfurter API.
- Generates signal by day-over-day move:
  - `long` if change >= threshold
  - `short` if change <= -threshold
  - `hold` otherwise
- Opens one paper position per pair when no position exists.
- Exits on:
  - opposite signal, or
  - max hold days reached

### Safety
- Bot refuses non-paper mode (`FOREX_MODE` must be `paper`).
- No interaction with equities DB (`data/trader.db`).

### Suggested Next Steps
1. Run forex bot for 1 week in paper.
2. Review `data/forex_trader.db` expectancy per pair.
3. Then tune pair list and threshold.
