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
