import sqlite3
import pandas as pd
import streamlit as st
from datetime import date
import time
import altair as alt

from src.trader.config import settings

DB_PATH = "data/trader.db"

st.set_page_config(page_title="Sentiment Trader Dashboard", layout="wide")
st.title("📈 Sentiment Trader Dashboard")

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Controls")

refresh_seconds = st.sidebar.slider("Auto refresh (seconds)", 5, 300, 30)
selected_date = st.sidebar.date_input("Select date", value=date.today())

symbol_filter = st.sidebar.text_input("Symbol filter (blank = all)", value="").strip().upper()
decision_filter = st.sidebar.multiselect(
    "Signal decision filter",
    options=["buy", "sell", "hold"],
    default=["buy", "sell", "hold"],
)

max_rows = st.sidebar.slider("Max rows", 50, 2000, 200)

st.sidebar.divider()
st.sidebar.caption(f"DB: {DB_PATH}")

# =========================
# Helpers
# =========================
def load_df(sql: str, params=()):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df

# =========================
# Load data for selected day
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

# Apply filters
if symbol_filter:
    if not signals.empty and "symbol" in signals.columns:
        signals = signals[signals["symbol"].str.upper() == symbol_filter]
    if not trades.empty and "symbol" in trades.columns:
        trades = trades[trades["symbol"].str.upper() == symbol_filter]

if not signals.empty and "decision" in signals.columns:
    signals = signals[signals["decision"].isin(decision_filter)]

signals = signals.head(max_rows)
trades = trades.head(max_rows)

# =========================
# Overview
# =========================
st.subheader("📊 Overview")

col1, col2, col3, col4 = st.columns(4)

signals_count = len(signals)
trades_count = len(trades)

gross_notional = float((trades["qty"] * trades["price"]).sum()) if not trades.empty else 0.0

daily_pnl = 0.0
if not trades.empty and "side" in trades.columns:
    # BUY = cash outflow (-), SELL = inflow (+)  (simple cashflow pnl proxy)
    daily_pnl = float(
        trades.apply(
            lambda r: r["qty"] * r["price"] * (1 if str(r["side"]).lower() == "sell" else -1),
            axis=1,
        ).sum()
    )

col1.metric("Signals", signals_count)
col2.metric("Trades", trades_count)
col3.metric("Gross Notional ($)", f"{gross_notional:,.2f}")
col4.metric("Daily PnL ($)", f"{daily_pnl:,.2f}")

# Kill switch indicator (based on selected day’s pnl proxy)
if daily_pnl <= -settings.daily_loss_limit_usd:
    st.error("🚨 KILL SWITCH ACTIVE — Trading Halted (based on selected day’s PnL proxy)")
else:
    st.success("🟢 Trading Enabled (based on selected day’s PnL proxy)")

# =========================
# Tabs: Signals / Trades / Charts
# =========================
tab_signals, tab_trades, tab_charts = st.tabs(["🧠 Signals", "💰 Trades", "📈 Charts"])

with tab_signals:
    st.subheader("🧠 Sentiment Signals")

    if signals.empty:
        st.info("No signals for this date/filter.")
    else:
        # Color-ish UX: show BUY/SELL/HOLD clearly via emoji column (simple + reliable)
        sig_view = signals.copy()
        if "decision" in sig_view.columns:
            sig_view["flag"] = sig_view["decision"].map(
                {"buy": "🟢 BUY", "sell": "🔴 SELL", "hold": "🟡 HOLD"}
            ).fillna(sig_view["decision"])

            # Move flag to front
            cols = ["flag"] + [c for c in sig_view.columns if c != "flag"]
            sig_view = sig_view[cols]

        st.dataframe(sig_view, use_container_width=True, height=320)

with tab_trades:
    st.subheader("💰 Trades")

    if trades.empty:
        st.info("No trades for this date/filter.")
    else:
        tr_view = trades.copy()
        if "side" in tr_view.columns:
            tr_view["flag"] = tr_view["side"].str.lower().map(
                {"buy": "🟢 BUY", "sell": "🔴 SELL"}
            ).fillna(tr_view["side"])
            cols = ["flag"] + [c for c in tr_view.columns if c != "flag"]
            tr_view = tr_view[cols]

        st.dataframe(tr_view, use_container_width=True, height=320)

with tab_charts:
    st.subheader("📈 Charts")

    # --- Sentiment chart + trade markers (if available) ---
    if not signals.empty and "timestamp" in signals.columns and "sentiment" in signals.columns:
        sig = signals.copy()
        sig["timestamp"] = pd.to_datetime(sig["timestamp"])

        line = alt.Chart(sig).mark_line().encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("sentiment:Q", title="Sentiment"),
            tooltip=["timestamp:T", "sentiment:Q", "decision:N", "symbol:N"],
        )

        chart = line

        if not trades.empty and "timestamp" in trades.columns:
            tr = trades.copy()
            tr["timestamp"] = pd.to_datetime(tr["timestamp"])

            pts = alt.Chart(tr).mark_point(size=140).encode(
                x="timestamp:T",
                y=alt.value(0),  # markers at baseline
                shape=alt.Shape("side:N"),
                tooltip=["timestamp:T", "symbol:N", "side:N", "qty:Q", "price:Q", "order_id:N"],
            )
            chart = line + pts

        st.altair_chart(chart, use_container_width=True)
        st.caption("Trade markers are plotted at y=0 (baseline) to avoid distorting sentiment scaling.")

    else:
        st.info("No sentiment data available for charting with these filters.")

    # --- Daily PnL history chart (across days) ---
    pnl_df = load_df(
        """
        SELECT DATE(timestamp) as day,
               SUM(qty * price * CASE WHEN LOWER(side)='sell' THEN 1 ELSE -1 END) as pnl
        FROM trades
        GROUP BY day
        ORDER BY day
        """
    )

    if not pnl_df.empty:
        st.subheader("📊 Daily PnL History (cashflow proxy)")
        st.line_chart(pnl_df.set_index("day"))
    else:
        st.info("No historical trades to compute PnL history.")

# =========================
# Auto-refresh
# =========================
time.sleep(refresh_seconds)
st.rerun()
