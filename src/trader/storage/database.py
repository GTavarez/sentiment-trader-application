import sqlite3
from pathlib import Path

DB_PATH = Path("data/trader.db")


def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        symbol TEXT,
        sentiment REAL,
        decision TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        symbol TEXT,
        side TEXT,
        qty INTEGER,
        price REAL,
        sentiment REAL,
        order_id TEXT
    )
    """)

    conn.commit()
    conn.close()
