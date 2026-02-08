from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from loguru import logger

DB_PATH = Path("data/trader.db")

def clear_db_positions(reason: str, symbols: list[str] | None = None) -> None:
    """
    Clears DB trade state (PAPER ONLY).
    Used when DB shows ghost positions but broker is clean.
    """
    if not DB_PATH.exists():
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    if symbols:
        placeholders = ", ".join("?" for _ in symbols)
        cur.execute(
            f"DELETE FROM trades WHERE UPPER(symbol) IN ({placeholders})",
            [s.upper() for s in symbols],
        )
    else:
        # You may choose to truncate trades OR mark as reconciled
        cur.execute("DELETE FROM trades")
    conn.commit()
    conn.close()

    logger.warning(
        f"AUTO-HEAL APPLIED — DB positions cleared ({reason}) "
        f"{'symbols=' + ','.join(symbols) if symbols else ''}"
    )


def add_broker_positions_to_db(
    reason: str,
    positions: list[dict],
) -> None:
    """
    Insert synthetic BUY trades for broker-held positions (PAPER ONLY).
    Each position dict: {symbol, qty, avg_entry_price}
    """
    if not DB_PATH.exists():
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()

    inserted = []
    for p in positions:
        symbol = str(p.get("symbol", "")).upper()
        qty = int(p.get("qty", 0))
        price = float(p.get("avg_entry_price", 0.0) or 0.0)
        if not symbol or qty <= 0:
            continue
        cur.execute(
            """
            INSERT INTO trades (timestamp, symbol, side, qty, price, sentiment, order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now, symbol, "buy", qty, price, 0.0, "AUTO_HEAL_BROKER_SYNC"),
        )
        inserted.append(symbol)

    conn.commit()
    conn.close()

    logger.warning(
        f"AUTO-HEAL APPLIED — DB positions rebuilt from broker ({reason}) "
        f"{'symbols=' + ','.join(inserted) if inserted else ''}"
    )
