import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


DB_PATH = Path("data/forex_trader.db")


def load_df(sql: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


st.set_page_config(page_title="Forex Paper Dashboard", layout="wide")
st.title("Forex Paper Dashboard")
st.caption("Data source: data/forex_trader.db")

if not DB_PATH.exists():
    st.error("Forex DB not found yet. Run `python forex_paper_bot.py` first.")
    st.stop()

positions = load_df(
    """
    SELECT pair, side, qty, entry_price, entry_ts
    FROM fx_positions
    ORDER BY pair
    """
)

closed = load_df(
    """
    SELECT id, timestamp, pair, side, qty, price, reason, pnl
    FROM fx_trades
    WHERE pnl IS NOT NULL
    ORDER BY timestamp DESC
    """
)

all_trades = load_df(
    """
    SELECT id, timestamp, pair, side, qty, price, reason, pnl
    FROM fx_trades
    ORDER BY timestamp DESC
    """
)

wins = int((closed["pnl"] > 0).sum()) if not closed.empty else 0
losses = int((closed["pnl"] < 0).sum()) if not closed.empty else 0
closed_n = wins + losses
win_rate = (wins / closed_n * 100.0) if closed_n > 0 else 0.0
total_pnl = float(closed["pnl"].sum()) if not closed.empty else 0.0
avg_win = float(closed.loc[closed["pnl"] > 0, "pnl"].mean()) if wins else 0.0
avg_loss = float(closed.loc[closed["pnl"] < 0, "pnl"].mean()) if losses else 0.0
wl_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0.0
expectancy = (total_pnl / closed_n) if closed_n > 0 else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Closed Trades", f"{closed_n}")
c2.metric("Wins / Losses", f"{wins} / {losses}")
c3.metric("Win Rate (%)", f"{win_rate:.1f}")
c4.metric("Total Realized PnL", f"{total_pnl:.2f}")
c5.metric("Win/Loss Ratio", f"{wl_ratio:.2f}")
c6.metric("Expectancy", f"{expectancy:.2f}")

st.subheader("Open Positions")
if positions.empty:
    st.info("No open forex positions.")
else:
    st.dataframe(positions, width="stretch")

st.subheader("Closed Trades (with PnL)")
if closed.empty:
    st.info("No closed trades yet.")
else:
    st.dataframe(closed, width="stretch")

st.subheader("All Trades")
if all_trades.empty:
    st.info("No trades logged yet.")
else:
    st.dataframe(all_trades, width="stretch")

pair_stats = load_df(
    """
    SELECT
        pair,
        COUNT(*) AS trades,
        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losses,
        ROUND(AVG(pnl), 4) AS expectancy,
        ROUND(SUM(pnl), 4) AS total_pnl
    FROM fx_trades
    WHERE pnl IS NOT NULL
    GROUP BY pair
    ORDER BY total_pnl ASC
    """
)

st.subheader("Pair Stats")
if pair_stats.empty:
    st.info("No pair stats yet.")
else:
    st.dataframe(pair_stats, width="stretch")
