  # src/trader/state/broker_positions.py

import pandas as pd
from loguru import logger

from src.trader.brokers.alpaca import AlpacaBroker


def load_live_positions(
    broker: AlpacaBroker,
) -> pd.DataFrame:
    """
    Read-only helper.

    Returns the current LIVE broker positions from Alpaca
    in a normalized dataframe.

    Columns:
        symbol
        qty
        avg_price
        market_value

    This function:
    - DOES NOT place orders
    - DOES NOT cancel orders
    - DOES NOT modify DB
    - NEVER raises (returns empty DF on failure)
    """

    try:
        positions = broker.get_positions()
    except Exception as e:
        logger.error(f"[RECON] Failed to fetch broker positions: {e}")
        return pd.DataFrame(
            columns=["symbol", "qty", "avg_price", "market_value"]
        )

    rows = []

    for p in positions:
        try:
            qty = float(p.qty)
            avg_price = float(getattr(p, "avg_entry_price", 0.0))
            market_value = float(getattr(p, "market_value", 0.0))
        except Exception:
            # Defensive: Alpaca sometimes returns strings
            qty = float(p.qty) if p.qty is not None else 0.0
            avg_price = float(p.avg_entry_price) if p.avg_entry_price else 0.0
            market_value = qty * avg_price

        rows.append(
            {
                "symbol": str(p.symbol).upper(),
                "qty": qty,
                "avg_price": avg_price,
                "market_value": market_value,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["symbol", "qty", "avg_price", "market_value"]
        )

    df = pd.DataFrame(rows)

    # Sort for deterministic display & joins
    df = df.sort_values("symbol").reset_index(drop=True)

    return df
