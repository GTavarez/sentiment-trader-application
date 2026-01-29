from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from loguru import logger

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def fetch_daily_bars_alpaca(
    api_key: str,
    secret_key: str,
    symbols: list[str],
    days: int = 60,
) -> pd.DataFrame:
    """
    Fetch daily close prices for symbols from Alpaca Market Data.
    Returns a DataFrame with columns: [timestamp, symbol, close]
    """

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    client = StockHistoricalDataClient(api_key, secret_key)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed= "iex",
        adjustment="raw",
    )

    bars = client.get_stock_bars(request)
    if bars is None or bars.data is None:
        return pd.DataFrame(columns=["timestamp", "symbol", "close"])

    rows = []
    # bars.data is dict: symbol -> list[Bar]
    for sym, bar_list in bars.data.items():
        for b in bar_list:
            rows.append(
                {
                    "timestamp": pd.to_datetime(b.timestamp),
                    "symbol": sym,
                    "close": float(b.close),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["symbol", "timestamp"])
    logger.info(f"Fetched {len(df)} daily bars for {len(symbols)} symbols")
    return df
