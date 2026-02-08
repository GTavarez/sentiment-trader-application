import sqlite3
from pathlib import Path

DB_PATH = Path("data/trader.db")

def clear_open_positions_from_db():
    """
    Remove ALL position-carrying trades.
    Used ONLY for PAPER auto-heal.
    """
    if not DB_PATH.exists():
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # This preserves trade history but zeroes net positions
    cur.execute("""
        DELETE FROM trades
        WHERE id IN (
            SELECT id FROM trades
        )
    """)

    conn.commit()
    conn.close()
