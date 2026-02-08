## Sentiment Trader Runbook

This project runs a sentiment-driven trading bot with safety gates, reconciliation, and a Streamlit dashboard.

### Quick Start (Paper)
1. Set environment variables in `.env`.
2. Run the bot:
```bash
python -m src.trader.main
```
3. Run the dashboard:
```bash
streamlit run streamlit_app.py
```

### Live Safety Gates
To enable live trading, all of these must be set:
```
TRADING_MODE=live
LIVE_TRADING_CONFIRM=true
LIVE_TRADING_CONFIRM_CODE=I_UNDERSTAND
READ_ONLY=false
```
If any are missing, the bot will exit before placing orders.

### Read-Only Live Test
Use live keys with no orders:
```
TRADING_MODE=live
LIVE_TRADING_CONFIRM=true
LIVE_TRADING_CONFIRM_CODE=I_UNDERSTAND
READ_ONLY=true
```

### Common Controls
- `ALLOW_SCALE_IN=true` to add to existing positions
- `TAKE_PROFIT_PCT` and `STOP_LOSS_PCT` for exits
- `TRAILING_STOP_ENABLED=true` and `TRAILING_STOP_PCT` for trailing exits
- `TREND_FILTER_ENABLED=true` and `VOLATILITY_FILTER_ENABLED=true` for filters

### Reconciliation & Recovery
- Streamlit includes:
  - Clear halt state
  - Rebuild DB from broker positions (admin)
- If reconciliation mismatches occur, the broker is the source of truth.

### Recommended Live Ramp
1. 1-2 days read-only live
2. Tiny caps: `MAX_POSITION_USD=25` and `MAX_SYMBOL_EXPOSURE_USD=50`
3. Increase slowly after consistent performance
