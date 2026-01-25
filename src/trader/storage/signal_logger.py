from datetime import datetime, timezone
from src.trader.storage.database import get_connection


def log_signal(symbol: str, sentiment: float, decision: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO signals (timestamp, symbol, sentiment, decision)
        VALUES (?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            symbol,
            sentiment,
            decision,
        ),
    )

    conn.commit()
    conn.close()
