from datetime import datetime, timezone
from src.trader.storage.database import get_connection


def log_trade(
    symbol: str,
    side: str,
    qty: int,
    price: float,
    sentiment: float,
    order_id: str,
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO trades
        (timestamp, symbol, side, qty, price, sentiment, order_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            symbol,
            side,
            qty,
            price,
            sentiment,
            str(order_id),
        ),
    )

    conn.commit()
    conn.close()


def log_order_attempt(
    symbol: str,
    side: str,
    qty: int,
    price: float | None,
    sentiment: float | None,
    order_id: str,
    status: str,
    reason: str = "",
):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO order_attempts
        (timestamp, symbol, side, qty, price, sentiment, order_id, status, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            symbol,
            side,
            int(qty),
            float(price) if price is not None else None,
            float(sentiment) if sentiment is not None else None,
            str(order_id),
            str(status),
            str(reason),
        ),
    )

    conn.commit()
    conn.close()
