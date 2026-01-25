import sqlite3
from pathlib import Path
from datetime import date
import time

import pandas as pd
import streamlit as st
import altair as alt


# =========================
# DB SETUP (cloud-safe)
# =========================
DB_PATH = Path("data/trader.db")

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def ensure_tables():
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

ensure_tables()


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Sentiment Trader Dashboard",
    layout="wide",
)

st.title("📈 Sentiment Trader Dashboard")


# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("Controls")

refresh_seconds = st.sidebar.slider("Auto refresh (seconds)", 5, 300, 30)
selected_date = st.sidebar.date_input("Select date", value=date.today())
symbol_filter = st.sidebar.text_input("Symbol filter (blank = all)", "").strip().upper()

decision_filter = st.sidebar.multiselect(
    "Signal decision filter",
    options=["buy", "sell", "hold"],
    default=["buy", "sell", "hold"],
)

max_rows = st.sidebar.slider("Max rows", 50, 2000, 200)

st.sidebar.divider()
st.sidebar.caption(f"DB: {DB_PATH}")


# =========================
# HELPERS
# =========================
def load_df(sql: str, params=()):
    conn = get_connection()
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


# =========================
# LOAD DATA
# =========================
signals = load_df(
    """
    SELECT * FROM signals
    WHERE DATE(timestamp) = ?
    ORDER BY timestamp DESC
    """,
    (selected_date.isoformat(),),
)

trades = load_df(
    """
    SELECT * FROM trades
    WHERE DATE(timestamp) = ?
    ORDER BY timestamp DESC
    """,
    (selected_date.isoformat(),),
)


# =========================
# APPLY FILTERS
# =========================
if symbol_filter:
    if not signals.empty:
        signals = signals[signals["symbol"].str.upper() == symbol_filter]
    if not trades.empty:
        trades = trades[trades["symbol"].str.upper() == symbol_filter]

if not signals.empty:
    signals = signals[signals["decision"].isin(decision_filter)]

signals = signals.head(max_rows)
trades = trades.head(max_rows)


# =========================
# OVERVIEW
# =========================
st.subheader("📊 Overview")

col1, col2, col3, col4 = st.columns(4)

signals_count = len(signals)
trades_count = len(trades)

gross_notional = (
    float((trades["qty"] * trades["price"]).sum())
    if not trades.empty else 0.0
)

daily_pnl = 0.0
if not trades.empty:
    daily_pnl = float(
        trades.apply(
            lambda r: r["qty"] * r["price"] *
            (1 if str(r["side"]).lower() == "sell" else -1),
            axis=1,
        ).sum()
    )

col1.metric("Signals", signals_count)
col2.metric("Trades", trades_count)
col3.metric("Gross Notional ($)", f"{gross_notional:,.2f}")
col4.metric("Daily PnL ($)", f"{daily_pnl:,.2f}")

if daily_pnl <= -300:
    st.error("🚨 KILL SWITCH ACTIVE — Trading Halted")
else:
    st.success("🟢 Trading Enabled")


# =========================
# TABS
# =========================
tab_signals, tab_trades, tab_charts = st.tabs(
    ["🧠 Signals", "💰 Trades", "📈 Charts"]
)

with tab_signals:
    if signals.empty:
        st.info("No signals for this date/filter.")
    else:
        st.dataframe(signals, use_container_width=True)

with tab_trades:
    if trades.empty:
        st.info("No trades for this date/filter.")
    else:
        st.dataframe(trades, use_container_width=True)

with tab_charts:
    if not signals.empty:
        sig = signals.copy()
        sig["timestamp"] = pd.to_datetime(sig["timestamp"])

        line = alt.Chart(sig).mark_line().encode(
            x="timestamp:T",
            y="sentiment:Q",
            tooltip=["timestamp:T", "sentiment:Q", "decision:N", "symbol:N"],
        )

        chart = line

        if not trades.empty:
            tr = trades.copy()
            tr["timestamp"] = pd.to_datetime(tr["timestamp"])

            pts = alt.Chart(tr).mark_point(size=120).encode(
                x="timestamp:T",
                y=alt.value(0),
                shape="side:N",
                tooltip=["timestamp:T", "symbol:N", "side:N", "qty:Q", "price:Q"],
            )

            chart = line + pts

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No sentiment data available.")


# =========================
# AUTO-REFRESH (LAST LINE)
# =========================
time.sleep(refresh_seconds)
st.rerun()
