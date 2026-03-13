import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


EQ_DB = Path("data/trader.db")
FX_DB = Path("data/forex_trader.db")


def load_df(db_path: Path, sql: str) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


def eq_realized_rows() -> pd.DataFrame:
    trades = load_df(
        EQ_DB,
        """
        SELECT timestamp, symbol, side, qty, price
        FROM trades
        ORDER BY timestamp ASC
        """,
    )
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "pnl"])

    trades["side"] = trades["side"].astype(str).str.lower()
    trades["symbol"] = trades["symbol"].astype(str).str.upper()
    trades["qty"] = trades["qty"].astype(int)
    trades["price"] = trades["price"].astype(float)

    fifo: dict[str, list[list[float]]] = {}
    out: list[dict] = []

    for _, r in trades.iterrows():
        sym = str(r["symbol"])
        side = str(r["side"])
        qty = int(r["qty"])
        px = float(r["price"])
        ts = str(r["timestamp"])

        fifo.setdefault(sym, [])
        if side == "buy":
            fifo[sym].append([qty, px])
            continue

        if side != "sell":
            continue

        rem = qty
        while rem > 0 and fifo[sym]:
            lot_qty, lot_px = fifo[sym][0]
            m = min(rem, int(lot_qty))
            pnl = (px - float(lot_px)) * m
            out.append({"timestamp": ts, "symbol": sym, "pnl": float(pnl)})
            lot_qty = int(lot_qty) - m
            rem -= m
            if lot_qty <= 0:
                fifo[sym].pop(0)
            else:
                fifo[sym][0][0] = lot_qty

    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "pnl"])
    return df[df["pnl"] != 0].copy()


def summarize_pnl(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "closed": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "ratio": 0.0,
            "expectancy": 0.0,
            "total_pnl": 0.0,
        }
    vals = df["pnl"].astype(float)
    wins = int((vals > 0).sum())
    losses = int((vals < 0).sum())
    closed = wins + losses
    avg_win = float(vals[vals > 0].mean()) if wins else 0.0
    avg_loss = float(vals[vals < 0].mean()) if losses else 0.0
    win_rate = (wins / closed * 100.0) if closed else 0.0
    ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0.0
    total_pnl = float(vals.sum())
    expectancy = (total_pnl / closed) if closed else 0.0
    return {
        "closed": closed,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "ratio": ratio,
        "expectancy": expectancy,
        "total_pnl": total_pnl,
    }


st.set_page_config(page_title="Equity vs Forex Comparison", layout="wide")
st.title("Equity vs Forex Comparison")
st.caption("Compares realized closed-trade performance across both bots.")

eq_rows = eq_realized_rows()
fx_rows = load_df(
    FX_DB,
    """
    SELECT timestamp, pair AS symbol, pnl
    FROM fx_trades
    WHERE pnl IS NOT NULL
    ORDER BY timestamp ASC
    """,
)

eq_stats = summarize_pnl(eq_rows)
fx_stats = summarize_pnl(fx_rows)

st.subheader("Headline Metrics")
c1, c2 = st.columns(2)
with c1:
    st.markdown("### Equities")
    st.metric("Closed Trades", f"{eq_stats['closed']}")
    st.metric("Wins / Losses", f"{eq_stats['wins']} / {eq_stats['losses']}")
    st.metric("Win Rate (%)", f"{eq_stats['win_rate']:.1f}")
    st.metric("Avg Win / Avg Loss", f"{eq_stats['avg_win']:.2f} / {eq_stats['avg_loss']:.2f}")
    st.metric("Win/Loss Ratio", f"{eq_stats['ratio']:.2f}")
    st.metric("Expectancy", f"{eq_stats['expectancy']:.2f}")
    st.metric("Total Realized PnL", f"{eq_stats['total_pnl']:.2f}")

with c2:
    st.markdown("### Forex")
    st.metric("Closed Trades", f"{fx_stats['closed']}")
    st.metric("Wins / Losses", f"{fx_stats['wins']} / {fx_stats['losses']}")
    st.metric("Win Rate (%)", f"{fx_stats['win_rate']:.1f}")
    st.metric("Avg Win / Avg Loss", f"{fx_stats['avg_win']:.2f} / {fx_stats['avg_loss']:.2f}")
    st.metric("Win/Loss Ratio", f"{fx_stats['ratio']:.2f}")
    st.metric("Expectancy", f"{fx_stats['expectancy']:.2f}")
    st.metric("Total Realized PnL", f"{fx_stats['total_pnl']:.2f}")

st.subheader("Per Instrument Performance")
lcol, rcol = st.columns(2)

with lcol:
    st.markdown("#### Equities by Symbol")
    if eq_rows.empty:
        st.info("No realized equity trades yet.")
    else:
        eq_pair = (
            eq_rows.groupby("symbol", as_index=False)
            .agg(
                trades=("pnl", "count"),
                wins=("pnl", lambda s: int((s > 0).sum())),
                losses=("pnl", lambda s: int((s < 0).sum())),
                expectancy=("pnl", "mean"),
                total_pnl=("pnl", "sum"),
            )
            .sort_values("total_pnl", ascending=True)
        )
        eq_pair["win_rate_pct"] = (eq_pair["wins"] / eq_pair["trades"] * 100.0).round(1)
        st.dataframe(eq_pair, width="stretch")

with rcol:
    st.markdown("#### Forex by Pair")
    if fx_rows.empty:
        st.info("No realized forex trades yet.")
    else:
        fx_pair = (
            fx_rows.groupby("symbol", as_index=False)
            .agg(
                trades=("pnl", "count"),
                wins=("pnl", lambda s: int((s > 0).sum())),
                losses=("pnl", lambda s: int((s < 0).sum())),
                expectancy=("pnl", "mean"),
                total_pnl=("pnl", "sum"),
            )
            .sort_values("total_pnl", ascending=True)
        )
        fx_pair["win_rate_pct"] = (fx_pair["wins"] / fx_pair["trades"] * 100.0).round(1)
        st.dataframe(fx_pair, width="stretch")

st.subheader("Recent Realized Trades")
t1, t2 = st.columns(2)
with t1:
    st.markdown("#### Equities (Latest 50)")
    if eq_rows.empty:
        st.info("No realized equity rows yet.")
    else:
        st.dataframe(eq_rows.sort_values("timestamp", ascending=False).head(50), width="stretch")
with t2:
    st.markdown("#### Forex (Latest 50)")
    if fx_rows.empty:
        st.info("No realized forex rows yet.")
    else:
        st.dataframe(fx_rows.sort_values("timestamp", ascending=False).head(50), width="stretch")
