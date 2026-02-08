import sqlite3
from pathlib import Path
from datetime import date
import time

import pandas as pd
import streamlit as st
import altair as alt

from src.trader.config import settings
from src.trader.backtest.price_loader import fetch_daily_bars_alpaca
from src.trader.backtest.backtester import run_backtest_with_prices
from src.trader.backtest.metrics import compute_metrics
from src.trader.backtest.portfolio_backtester import run_portfolio_backtest
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
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=True)
refresh_seconds = st.sidebar.slider("Auto refresh (seconds)", 5, 300, 30)
selected_date = st.sidebar.date_input("Select date", value=date.today())
symbol_filter = st.sidebar.text_input("Symbol filter (blank = all)", "").strip().upper()
st.subheader("Backtest Costs")

slippage_bps = st.slider("Slippage (bps)", 0, 50, 2)          # 2 bps default
fee_per_trade = st.number_input("Fee per trade ($)", 0.0, 10.0, 0.50, step=0.10)

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

symbols = sorted(
    set(
        load_df("SELECT DISTINCT symbol FROM signals")["symbol"].tolist()
        + load_df("SELECT DISTINCT symbol FROM trades")["symbol"].tolist()
    )
)

symbol_filter = st.sidebar.selectbox(
    "Symbol",
    options=["ALL"] + symbols,
    index=0,
)

# =========================
# APPLY FILTERS
# =========================
if symbol_filter != "ALL":
    signals = signals[signals["symbol"] == symbol_filter]
    trades = trades[trades["symbol"] == symbol_filter]


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
    st.error("🚨 KILL SWITCH ACTIVE — Trading Halted", icon="🚨")
elif daily_pnl <= -150:
    st.warning("⚠️ Approaching Daily Loss Limit")
else:
    st.success("🟢 Trading Enabled")



# =========================
# TABS
# =========================
tab_signals, tab_trades, tab_charts, tab_backtest = st.tabs(
    ["🧠 Signals", "💰 Trades", "📈 Charts", "🧪 Backtest"]
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
with tab_backtest:
    st.subheader("🧪 Backtest (Real Prices)")

    bt_days = st.slider("Lookback days", 30, 365, 120)
    bt_trade_size = st.number_input("Trade size ($)", 50, 5000, 500)
    bt_starting_cash = st.number_input("Starting cash ($)", 1000, 500000, 100000)

    bt_signals = load_df(
    """
    SELECT * FROM signals
    WHERE timestamp >= DATE('now', ?)
    ORDER BY timestamp ASC
    """,
    (f"-{bt_days} day",),
)

    if bt_signals.empty:
         st.info("No signals available for backtesting.")
         st.stop()

    else:
        symbols = sorted(bt_signals["symbol"].unique().tolist())

        st.caption(f"Fetching daily bars for: {', '.join(symbols)}")
        hist = (
            alt.Chart(bt_signals)
            .mark_bar()
            .encode(
                 alt.X("sentiment:Q", bin=alt.Bin(maxbins=50)),
                  y="count()",
    )
)

        st.altair_chart(hist, use_container_width=True)

    prices = fetch_daily_bars_alpaca(
            api_key=st.secrets["ALPACA_API_KEY"],
            secret_key=st.secrets["ALPACA_SECRET_KEY"],
            symbols=symbols,
            days=int(bt_days),
        )

    trades_df = pd.DataFrame()
    equity_df = pd.DataFrame()
    summary = {}
        
    if prices.empty:
            st.error("No historical prices returned. Check Alpaca market data access.")
    else:
            trades_df, equity_df, summary = run_backtest_with_prices(
                signals=bt_signals,
                prices=prices,
                buy_threshold=settings.buy_threshold,
                sell_threshold=settings.sell_threshold,
                starting_cash=float(bt_starting_cash),
                trade_size_usd=float(bt_trade_size),
                slippage_bps=float(slippage_bps),
                fee_per_trade=float(fee_per_trade),
            )
            st.write("### Summary")
            st.json(summary)
            st.write("Equity rows:", len(equity_df))
            st.write("Trades rows:", len(trades_df))
    if not trades_df.empty:
            st.write("Side counts:", trades_df["side"].value_counts())

    metrics = compute_metrics(equity_df)
    if metrics:
            st.subheader("📊 Backtest Performance")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe Ratio", metrics["sharpe"])
            c2.metric("Max Drawdown", f'{metrics["max_drawdown_pct"]}%')
            c3.metric("Total Return", f'{metrics["total_return_pct"]}%')
            c4.metric("Trades", summary["trades"])

    if not trades_df.empty:
            st.write("### Trades")
            st.dataframe(trades_df, use_container_width=True)

    if not equity_df.empty:
            st.subheader("📈 Portfolio Equity Curve")
            ymin = equity_df["equity"].min() * 0.999
            ymax = equity_df["equity"].max() * 1.001

            st.line_chart(
            equity_df.set_index("timestamp")["equity"],
            height=350,
            use_container_width=True,
            )

            st.caption(f"Equity range: {equity_df['equity'].min():,.2f} → {equity_df['equity'].max():,.2f}")
    else:
            st.info("No portfolio equity generated.")

    if not trades_df.empty:
            st.subheader("### Backtest Summary")
    else:
            st.info("No trades triggered by thresholds.")
            st.write("Equity DF preview")
            st.write(equity_df.head())
            st.write(equity_df.tail())
            st.write("Equity rows:", len(equity_df))
            st.write(equity_df.head())


# =========================
# Portfolio Backtest
# =========================
equity_df, summary, trades_df, *_ = run_portfolio_backtest(
    signals=bt_signals,
    prices=prices,
    buy_threshold=settings.buy_threshold,
    sell_threshold=settings.sell_threshold,
    starting_cash=100_000,
    trade_risk_pct=0.05,
)
if equity_df.empty:
    st.warning("No portfolio equity generated — check signals & prices.")
else:
    st.line_chart(equity_df.set_index("timestamp")["equity"])

result = run_portfolio_backtest(
    signals=bt_signals,
    prices=prices,
    buy_threshold=settings.buy_threshold,
    sell_threshold=settings.sell_threshold,
    starting_cash=100_000,
    trade_risk_pct=0.05,
)

st.write("Returned type:", type(result))
st.write("Returned length:", len(result))
st.write(result)


if equity_df.empty:
    st.warning("No portfolio equity generated — check signals & prices.")
else:
    # ---- Equity Curve ----
    st.subheader("📈 Portfolio Equity Curve")
    st.line_chart(
        equity_df.set_index("timestamp")["equity"],
        height=350,
    )

       

    st.subheader("📊 Backtest Performance")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    c2.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
    c3.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
    c4.metric("Trades", metrics["trades"])

    # ---- Summary ----
    st.subheader("💼 Portfolio Summary")
    c1, c2 = st.columns(2)
    c1.metric("Ending Equity", f"${summary['ending_equity']:,.2f}")
    c2.metric("Net PnL", f"${summary['net_pnl']:,.2f}")


if not signals.empty:
    sig = signals.copy()
    sig["timestamp"] = pd.to_datetime(sig["timestamp"])

    base = alt.Chart(sig)

    sentiment_line = base.mark_line().encode(
        x="timestamp:T",
        y=alt.Y("sentiment:Q", scale=alt.Scale(domain=[-1, 1])),
        tooltip=["timestamp:T", "sentiment:Q", "decision:N", "symbol:N"],
    )

    chart = sentiment_line

    if not trades.empty:
        tr = trades.copy()
        tr["timestamp"] = pd.to_datetime(tr["timestamp"])

        # Match trade to nearest sentiment timestamp
        tr = pd.merge_asof(
            tr.sort_values("timestamp"),
            sig[["timestamp", "sentiment"]].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
        )

        trade_points = alt.Chart(tr).mark_point(
            size=140,
            filled=True
        ).encode(
            x="timestamp:T",
            y="sentiment:Q",
            color=alt.Color(
                "side:N",
                scale=alt.Scale(domain=["buy", "sell"], range=["green", "red"])
            ),
            shape=alt.Shape(
                "side:N",
                scale=alt.Scale(domain=["buy", "sell"], range=["triangle-up", "triangle-down"])
            ),
            tooltip=["timestamp:T", "side:N", "qty:Q", "price:Q"],
        )

        chart = sentiment_line + trade_points

    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No signals available for charting.")



# =========================
# AUTO-REFRESH (LAST LINE)
# =========================
if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()

