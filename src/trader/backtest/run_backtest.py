from price_loader import load_prices
from backtester import run_backtest
from metrics import sharpe_ratio, max_drawdown


prices = load_prices("AAPL", "2022-01-01", "2024-01-01")

# TEMP mock signal (replace later with sentiment logic)
signals = prices.index.to_series().apply(
    lambda _: "buy"
)

equity = run_backtest(prices, signals)
returns = equity.pct_change().dropna()

print("Sharpe:", sharpe_ratio(returns))
print("Max Drawdown:", max_drawdown(equity))
