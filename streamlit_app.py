import streamlit as st
from datetime import datetime, timedelta
import json
import sqlite3
import pandas as pd
from pathlib import Path
from src.trader.config import settings
from src.trader.brokers.alpaca import AlpacaBroker
from src.trader.state.streaks import load_streaks
from src.trader.risk.pnl import calculate_daily_pnl
from src.trader.state.reconciliation import reconcile_positions
from src.trader.state.db_positions import load_latest_positions
from src.trader.state.halt_state import (
    load_halt_state,
    write_unblock_ack,
    compute_fingerprint,
    clear_halt,
)
from src.trader.state.symbols import load_symbols, save_symbols




DB_PATH = Path("data/trader.db")
ENV_PATH = Path(".env")
CHECKLIST_PATH = Path("data/phase_checklist.json")
HEARTBEAT_PATH = Path("data/heartbeat.txt")
     
def load_df(sql: str, params=()):
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df

BLOCK_FILE = Path("data/block_reasons.json")

def load_block_reasons():
    if not BLOCK_FILE.exists():
        return {}
    return json.loads(BLOCK_FILE.read_text())


def rebuild_trades_from_broker(broker: AlpacaBroker) -> list[tuple[str, int]]:
    positions = broker.get_positions()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM trades")
    now = datetime.utcnow().isoformat()
    rebuilt = []
    for p in positions:
        sym = getattr(p, "symbol", None)
        qty = int(getattr(p, "qty", 0))
        price = float(getattr(p, "avg_entry_price", 0.0) or 0.0)
        if not sym or qty <= 0:
            continue
        cur.execute(
            """
            INSERT INTO trades (timestamp, symbol, side, qty, price, sentiment, order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now, sym, "buy", qty, price, 0.0, "REBUILD_FROM_BROKER"),
        )
        rebuilt.append((sym, qty))
    conn.commit()
    conn.close()
    return rebuilt



def load_latest_signals():
    if not DB_PATH.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT s.*
        FROM signals s
        INNER JOIN (
            SELECT symbol, MAX(timestamp) AS ts
            FROM signals
            GROUP BY symbol
        ) latest
        ON s.symbol = latest.symbol AND s.timestamp = latest.ts
        ORDER BY s.symbol
        """,
        conn,
    )
    conn.close()
    return df
COOLDOWN_FILE = Path("data/cooldowns.json")

def load_cooldowns():
    if not COOLDOWN_FILE.exists():
        return {}
    raw = json.loads(COOLDOWN_FILE.read_text())
    return {k: datetime.fromisoformat(v) for k, v in raw.items()}


def load_phase_checklist() -> dict:
    if not CHECKLIST_PATH.exists():
        return {}
    try:
        return json.loads(CHECKLIST_PATH.read_text())
    except Exception:
        return {}


def save_phase_checklist(state: dict) -> None:
    CHECKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKLIST_PATH.write_text(json.dumps(state, indent=2))


def compute_realized_pnl(trades_df: pd.DataFrame):
    """
    FIFO realized PnL for long-only trades.
    Returns: total_pnl, wins, losses, avg_win, avg_loss, pnl_by_time_df
    """
    if trades_df.empty:
        return 0.0, 0, 0, 0.0, 0.0, pd.DataFrame(columns=["timestamp", "cum_pnl"])

    df = trades_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["side"] = df["side"].astype(str).str.lower()
    df = df.sort_values("timestamp")

    fifo = {}
    realized = []

    for _, row in df.iterrows():
        symbol = str(row["symbol"]).upper()
        side = row["side"]
        qty = int(row["qty"])
        price = float(row["price"])
        ts = row["timestamp"]

        if symbol not in fifo:
            fifo[symbol] = []

        if side == "buy":
            fifo[symbol].append([qty, price])
        elif side == "sell":
            remaining = qty
            while remaining > 0 and fifo[symbol]:
                lot_qty, lot_price = fifo[symbol][0]
                match_qty = min(remaining, lot_qty)
                pnl = (price - lot_price) * match_qty
                realized.append({"timestamp": ts, "pnl": pnl})
                lot_qty -= match_qty
                remaining -= match_qty
                if lot_qty == 0:
                    fifo[symbol].pop(0)
                else:
                    fifo[symbol][0][0] = lot_qty

    if not realized:
        return 0.0, 0, 0, 0.0, 0.0, pd.DataFrame(columns=["timestamp", "cum_pnl"])

    pnl_df = pd.DataFrame(realized).sort_values("timestamp")
    pnl_df["cum_pnl"] = pnl_df["pnl"].cumsum()

    wins = (pnl_df["pnl"] > 0).sum()
    losses = (pnl_df["pnl"] < 0).sum()
    avg_win = pnl_df.loc[pnl_df["pnl"] > 0, "pnl"].mean() if wins > 0 else 0.0
    avg_loss = pnl_df.loc[pnl_df["pnl"] < 0, "pnl"].mean() if losses > 0 else 0.0
    total_pnl = float(pnl_df["pnl"].sum())

    return total_pnl, int(wins), int(losses), float(avg_win), float(avg_loss), pnl_df[["timestamp", "cum_pnl"]]


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Sentiment Trader — Live Monitor",
    layout="wide",
)

st.title("📊 Sentiment Trader — Live Monitor")
paper_mode = settings.trading_mode.lower() != "live"
api_key = settings.alpaca_api_key
secret_key = settings.alpaca_secret_key
if paper_mode and settings.alpaca_api_key_paper and settings.alpaca_secret_key_paper:
    api_key = settings.alpaca_api_key_paper
    secret_key = settings.alpaca_secret_key_paper
if not paper_mode and settings.alpaca_api_key_live and settings.alpaca_secret_key_live:
    api_key = settings.alpaca_api_key_live
    secret_key = settings.alpaca_secret_key_live
if not api_key or not secret_key:
    st.warning("⚠️ Alpaca keys missing for current mode. Check .env *_PAPER/*_LIVE keys.")

# ----------------------------
# Bot Heartbeat
# ----------------------------
st.subheader("💓 Bot Heartbeat")
if HEARTBEAT_PATH.exists():
    hb = HEARTBEAT_PATH.read_text().strip()
    st.caption(hb if hb else "No heartbeat data.")
else:
    st.warning("No heartbeat file found yet.")

# ----------------------------
# Trading mode
# ----------------------------
mode = settings.trading_mode.upper()
if mode == "LIVE":
    st.error("🚨 LIVE TRADING ENABLED")
else:
    st.success("🧪 PAPER TRADING MODE")

# Live confirmation banner (visibility only)
if mode == "LIVE":
    confirm_ok = bool(settings.live_trading_confirm)
    code_ok = settings.live_trading_confirm_code.strip().upper() == "I_UNDERSTAND"
    if confirm_ok and code_ok:
        st.success("✅ Live confirmation flags are set (READY)")
    else:
        st.warning(
            "⚠️ Live confirmation flags are missing/invalid. "
            "Set LIVE_TRADING_CONFIRM=true and LIVE_TRADING_CONFIRM_CODE=I_UNDERSTAND."
        )

# ----------------------------
# Phase Checklist
# ----------------------------
st.subheader("✅ Phase Checklist")
st.caption("Check items as you complete them. Saved to data/phase_checklist.json.")

phase_items = [
    ("phase1_stability", "Phase 1 — Stability: bot runs daily without crashes"),
    ("phase1_recon", "Phase 1 — Reconciliation clean and halt/unblock flow tested"),
    ("phase1_alerts", "Phase 1 — Email alerts working"),
    ("phase2_paper", "Phase 2 — Paper trading for 2+ weeks with consistent data"),
    ("phase2_metrics", "Phase 2 — Track realized win rate, avg win/loss, drawdown"),
    ("phase3_edge", "Phase 3 — Identify which symbols/signals drive profits"),
    ("phase4_live_ro", "Phase 4 — Live read-only for 1–2 weeks"),
    ("phase5_tiny_live", "Phase 5 — Tiny live size with strict caps"),
    ("phase6_scale", "Phase 6 — Gradual size increase with safeguards"),
]

check_state = load_phase_checklist()
updated = False
for key, label in phase_items:
    current = bool(check_state.get(key, False))
    c1, c2 = st.columns([2, 1])
    new_val = c1.checkbox(label, value=current, key=f"chk_{key}")
    target_key = f"{key}_target"
    target_val = str(check_state.get(target_key, "") or "")
    new_target = c2.text_input("Target date", value=target_val, key=f"target_{key}")
    if new_val != current:
        check_state[key] = new_val
        updated = True
    if new_target != target_val:
        check_state[target_key] = new_target
        updated = True

if updated:
    save_phase_checklist(check_state)

# ----------------------------
# Go-Live Checklist
# ----------------------------
# Go-Live Day 1 preset (writes .env for next run)
st.subheader("🚦 Go-Live Day 1 Preset")
st.caption("Applies conservative caps to `.env` (requires restart to take effect).")
def _set_env_value(lines: list[str], key: str, value: str) -> list[str]:
    prefix = f"{key}="
    found = False
    out = []
    for line in lines:
        if line.startswith(prefix):
            out.append(f"{prefix}{value}\n")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{prefix}{value}\n")
    return out

if st.button("✅ Apply Day 1 Safe Caps", key="apply_day1_caps_btn"):
    env_path = Path(".env")
    try:
        lines = env_path.read_text().splitlines(keepends=True) if env_path.exists() else []
        lines = _set_env_value(lines, "MAX_POSITION_USD", "25")
        lines = _set_env_value(lines, "MAX_SYMBOL_EXPOSURE_USD", "50")
        lines = _set_env_value(lines, "DAILY_LOSS_LIMIT_USD", "50")
        env_path.write_text("".join(lines))
        st.success("Day 1 caps written to .env. Restart the app/bot to apply.")
    except Exception as e:
        st.error(f"Failed to update .env: {e}")

# Quick Mode Toggles
st.subheader("🧭 Quick Mode Toggles")
st.caption("Writes to `.env` (restart required).")
def _set_env_flag(key: str, value: str):
    env_path = Path(".env")
    lines = env_path.read_text().splitlines(keepends=True) if env_path.exists() else []
    lines = _set_env_value(lines, key, value)
    env_path.write_text("".join(lines))

c1, c2 = st.columns(2)
if c1.button("🧪 Switch to PAPER", key="switch_paper_btn"):
    try:
        _set_env_flag("TRADING_MODE", "paper")
        _set_env_flag("READ_ONLY", "false")
        st.success("Set TRADING_MODE=paper and READ_ONLY=false. Restart to apply.")
    except Exception as e:
        st.error(f"Failed to update .env: {e}")

if c2.button("🚨 Switch to LIVE (Read-Only)", key="switch_live_ro_btn"):
    try:
        _set_env_flag("TRADING_MODE", "live")
        _set_env_flag("READ_ONLY", "true")
        st.success("Set TRADING_MODE=live and READ_ONLY=true. Restart to apply.")
    except Exception as e:
        st.error(f"Failed to update .env: {e}")

if st.button("🚨 Switch to LIVE (Trading)", key="switch_live_btn"):
    st.warning("This enables LIVE trading. Make sure your account is funded.")
    try:
        _set_env_flag("TRADING_MODE", "live")
        _set_env_flag("READ_ONLY", "false")
        st.success("Set TRADING_MODE=live and READ_ONLY=false. Restart to apply.")
    except Exception as e:
        st.error(f"Failed to update .env: {e}")
# =========================
# PHASE 5.5 — LIVE EXPOSURE
# =========================
st.subheader("🧾 Live Exposure & Capital Utilization")

positions_df = load_latest_positions()
total_unrealized = 0.0

# If you have price data available from Alpaca daily bars, use it to estimate market value.
# We'll try to use `prices` from backtest tab if it exists, otherwise fallback to last trade price.
if not positions_df.empty:
    # Fallback: last trade price per symbol
    last_trade_px = load_df(
        """
        SELECT symbol, price, MAX(timestamp) as ts
        FROM trades
        GROUP BY symbol
        """
    )
    last_trade_px = last_trade_px[["symbol", "price"]].rename(columns={"price": "last_price"})

    positions_df = positions_df.merge(last_trade_px, on="symbol", how="left")
    positions_df["last_price"] = positions_df["last_price"].fillna(positions_df["avg_cost"])

    positions_df["market_value"] = positions_df["qty"] * positions_df["last_price"]
    positions_df["cost_basis"] = positions_df["qty"] * positions_df["avg_cost"]
    positions_df["unrealized_pnl"] = positions_df["market_value"] - positions_df["cost_basis"]

    total_market_value = float(positions_df["market_value"].sum())
    total_cost_basis = float(positions_df["cost_basis"].sum())
    total_unrealized = float(positions_df["unrealized_pnl"].sum())

    # Estimate "cash" from starting cash minus net flow (buys - sells)
    # This is an estimate based on DB trades only (works well for PAPER + this app).
    cash_flow = load_df(
        """
        SELECT side, qty, price
        FROM trades
        """
    )
    if cash_flow.empty:
        est_cash = 100000.0
    else:
        cash_flow["side"] = cash_flow["side"].astype(str).str.lower()
        cash_flow["signed_notional"] = cash_flow.apply(
            lambda r: (r["qty"] * r["price"]) * (-1 if r["side"] == "buy" else 1), axis=1
        )
        est_cash = float(100000.0 + cash_flow["signed_notional"].sum())

    est_equity = est_cash + total_market_value

    # Utilization vs max exposure (use your settings.max_position_usd and derived exposure caps)
    max_total_exposure = float(getattr(settings, "max_total_exposure_usd", settings.max_position_usd * 3))
    utilization = 0.0 if max_total_exposure <= 0 else min(1.0, total_market_value / max_total_exposure)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Est. Equity ($)", f"{est_equity:,.2f}")
    c2.metric("Est. Cash ($)", f"{est_cash:,.2f}")
    c3.metric("Deployed ($)", f"{total_market_value:,.2f}")
    c4.metric("Unrealized PnL ($)", f"{total_unrealized:,.2f}")

    st.progress(utilization, text=f"Capital utilization: {utilization*100:.1f}% of exposure cap (${max_total_exposure:,.0f})")

    st.write("### Current Positions (from DB trades)")
    st.dataframe(
        positions_df[["symbol", "qty", "avg_cost", "last_price", "market_value", "unrealized_pnl"]].sort_values("market_value", ascending=False),
        use_container_width=True,
    )
else:
    st.info("No open positions found in DB (trades table). Once trades are executed, positions will appear here.")

st.subheader("🧩 Exposure Caps")
c1, c2 = st.columns(2)
c1.metric("Max Symbol Exposure (Default $)", f"{settings.max_symbol_exposure_usd:,.2f}")
per_symbol_caps = settings.max_symbol_exposure_by_symbol or {}
if per_symbol_caps:
    caps_rows = [
        {"Symbol": k.upper(), "Max Exposure ($)": float(v)}
        for k, v in per_symbol_caps.items()
    ]
    st.dataframe(pd.DataFrame(caps_rows), use_container_width=True)
else:
    c2.info("No per-symbol caps set.")

# ----------------------------
# Symbols (editable)
# ----------------------------
st.subheader("🔤 Symbols")
current_symbols = load_symbols(settings.symbol_list)
symbols_text = ", ".join(current_symbols)
symbols_input = st.text_area(
    "Symbols (comma-separated)",
    value=symbols_text,
    height=80,
    key="symbols_input",
)
if st.button("💾 Save Symbols", key="save_symbols_btn"):
    parsed = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    if parsed:
        save_symbols(parsed)
        st.success(f"Saved {len(parsed)} symbols.")
    else:
        st.warning("No valid symbols found.")

if settings.max_symbols_per_run > 0:
    st.caption(f"Max symbols per run: {settings.max_symbols_per_run}")

# ----------------------------
# Broker (READ-ONLY)
# ----------------------------
broker = AlpacaBroker(
    api_key=api_key,
    secret_key=secret_key,
    paper=paper_mode,
)
try:
    key_prefix = settings.alpaca_api_key[:4]
    st.caption(f"Alpaca key prefix: {key_prefix}****")
except Exception:
    pass

# ----------------------------
# Account snapshot
# ----------------------------
st.subheader("💰 Account")

try:
    account = broker.get_account()
    equity = float(account.equity)
    daily_pnl = calculate_daily_pnl(broker)
    buying_power = float(getattr(account, "buying_power", 0.0) or 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity ($)", f"{equity:,.2f}")
    c2.metric("Daily PnL ($)", f"{daily_pnl:,.2f}")
    c3.metric("Updated", datetime.utcnow().strftime("%H:%M:%S UTC"))
    c4.metric("Buying Power ($)", f"{buying_power:,.2f}")
    if buying_power <= 0:
        st.warning("⚠️ Buying power is $0 — fund the live account to place trades.")
except Exception as e:
    st.error(f"Broker unavailable: {e}")
    st.stop()

# ----------------------------
# Today's Trades
# ----------------------------
st.subheader("🧾 Trades Today (UTC)")
start_utc = datetime.utcnow().strftime("%Y-%m-%dT00:00:00")
trades_today = load_df(
    """
    SELECT timestamp, symbol, side, qty, price, sentiment, order_id
    FROM trades
    WHERE timestamp >= ?
    ORDER BY timestamp DESC
    """,
    (start_utc,),
)
if trades_today.empty:
    st.info("No trades recorded today.")
else:
    st.dataframe(trades_today, use_container_width=True)

# ----------------------------
# Trades Last 7 Days
# ----------------------------
st.subheader("📅 Trades Last 7 Days (UTC)")
since_utc = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%dT00:00:00")
trades_7d = load_df(
    """
    SELECT timestamp, symbol, side, qty, price, sentiment, order_id
    FROM trades
    WHERE timestamp >= ?
    ORDER BY timestamp DESC
    """,
    (since_utc,),
)
if trades_7d.empty:
    st.info("No trades recorded in the last 7 days.")
else:
    st.dataframe(trades_7d, use_container_width=True)

# ----------------------------
# Performance Report
# ----------------------------
st.subheader("📈 Performance Report")
all_trades = load_df(
    """
    SELECT timestamp, symbol, side, qty, price, sentiment, order_id
    FROM trades
    ORDER BY timestamp ASC
    """
)
total_pnl, wins, losses, avg_win, avg_loss, pnl_curve = compute_realized_pnl(all_trades)
total_closed = wins + losses
win_rate = (wins / total_closed * 100.0) if total_closed > 0 else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Realized PnL ($)", f"{total_pnl:,.2f}")
c2.metric("Win Rate (%)", f"{win_rate:.1f}")
c3.metric("Wins", f"{wins}")
c4.metric("Avg Win ($)", f"{avg_win:,.2f}")
c5.metric("Avg Loss ($)", f"{avg_loss:,.2f}")

# Drawdown stats (realized)
if pnl_curve.empty:
    max_drawdown = 0.0
else:
    dd_df = pnl_curve.copy()
    dd_df["cum_pnl"] = dd_df["cum_pnl"].astype(float)
    running_max = dd_df["cum_pnl"].cummax()
    drawdown = dd_df["cum_pnl"] - running_max
    max_drawdown = float(drawdown.min())

st.metric("Max Drawdown ($)", f"{max_drawdown:,.2f}")

# ----------------------------
# Baseline Comparison (7d, size-aware proxy)
# ----------------------------
st.subheader("🧪 Baseline Comparison (7d Size-Aware Proxy)")
if trades_7d.empty:
    st.info("No trades in the last 7 days to compare.")
else:
    baseline_rows = []
    symbols_7d = trades_7d["symbol"].astype(str).str.upper().unique().tolist()
    for sym in symbols_7d:
        sym_trades = trades_7d[trades_7d["symbol"].astype(str).str.upper() == sym].copy()
        sym_trades["side"] = sym_trades["side"].astype(str).str.lower()
        buys = sym_trades[sym_trades["side"] == "buy"]
        if buys.empty:
            continue
        qty_sum = buys["qty"].astype(float).sum()
        if qty_sum <= 0:
            continue
        vwap = (buys["qty"].astype(float) * buys["price"].astype(float)).sum() / qty_sum
        try:
            current_price = float(broker.get_last_price(sym))
        except Exception:
            current_price = None
        if current_price is None:
            continue
        pnl = (current_price - vwap) * qty_sum
        pct = (current_price - vwap) / vwap * 100.0 if vwap > 0 else 0.0
        baseline_rows.append(
            {
                "Symbol": sym,
                "Qty (buys)": qty_sum,
                "VWAP Buy": vwap,
                "Current Price": current_price,
                "Baseline PnL": pnl,
                "Return (%)": pct,
            }
        )

    if not baseline_rows:
        st.info("Baseline data unavailable (missing current prices or buys).")
    else:
        baseline_df = pd.DataFrame(baseline_rows)
        st.dataframe(baseline_df, use_container_width=True)
        total_baseline = baseline_df["Baseline PnL"].sum()
        st.metric("Baseline PnL (size-aware)", f"{total_baseline:,.2f}")

# ----------------------------
# Performance Chart
# ----------------------------
st.subheader("📊 PnL Curves (Realized + Unrealized)")
if pnl_curve.empty:
    if total_unrealized != 0.0:
        now = datetime.utcnow()
        chart_df = pd.DataFrame(
            [
                {
                    "Time": now,
                    "Realized PnL": 0.0,
                    "Realized + Unrealized": total_unrealized,
                }
            ]
        ).set_index("Time")
        st.line_chart(chart_df)
        st.caption("Only unrealized PnL available (no completed sells yet).")
    else:
        st.info("No realized PnL yet (needs completed sells).")
else:
    chart_df = pnl_curve.rename(columns={"timestamp": "Time", "cum_pnl": "Realized PnL"})
    chart_df = chart_df.set_index("Time")
    if total_unrealized != 0.0:
        chart_df["Realized + Unrealized"] = chart_df["Realized PnL"]
        chart_df.loc[datetime.utcnow(), "Realized + Unrealized"] = (
            chart_df["Realized PnL"].iloc[-1] + total_unrealized
        )
    st.line_chart(chart_df)

# ----------------------------
# Open positions
# ----------------------------
st.subheader("📦 Open Positions")
positions = broker.get_positions()

if not positions:
    st.info("No open positions")
else:
    trail_label = (
        f"ON ({settings.trailing_stop_pct:.2%})"
        if settings.trailing_stop_enabled
        else "OFF"
    )
    st.dataframe(
        [
            {
                "Symbol": p.symbol,
                "Qty": p.qty,
                "Market Value": float(p.market_value),
                "Avg Entry": float(p.avg_entry_price),
                "Trailing Stop": trail_label,
            }
            for p in positions
        ],
        use_container_width=True,
    )

    rows = []
    total_unreal = 0.0
    wins = losses = 0
    for p in positions:
        qty = float(getattr(p, "qty", 0) or 0)
        avg_entry = float(getattr(p, "avg_entry_price", 0.0) or 0.0)
        current = float(getattr(p, "current_price", 0.0) or 0.0)
        if current <= 0 and qty > 0:
            current = float(getattr(p, "market_value", 0.0) or 0.0) / qty
        unreal = float(getattr(p, "unrealized_pl", 0.0) or 0.0)
        plpc = float(getattr(p, "unrealized_plpc", 0.0) or 0.0)
        total_unreal += unreal
        if unreal > 0:
            wins += 1
        elif unreal < 0:
            losses += 1
        rows.append(
            {
                "Symbol": p.symbol,
                "Qty": qty,
                "Avg Entry": avg_entry,
                "Current": current,
                "Unrealized PnL": unreal,
                "Return (%)": plpc * 100.0,
                "Status": "WIN" if unreal > 0 else ("LOSS" if unreal < 0 else "FLAT"),
            }
        )

    st.subheader("✅ Unrealized Wins (Broker)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Winning Positions", wins)
    c2.metric("Losing Positions", losses)
    c3.metric("Total Unrealized PnL", f"{total_unreal:,.2f}")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ----------------------------
# Sentiment streaks
# ----------------------------
st.subheader("🔥 Sentiment Streaks")

streaks = load_streaks()
if not streaks:
    st.info("No sentiment streaks recorded yet")
else:
    st.dataframe(
        [
            {
                "Symbol": sym,
                "Buy Streak": v.get("buy", 0),
                "Sell Streak": v.get("sell", 0),
            }
            for sym, v in streaks.items()
        ],
        use_container_width=True,
    )

# ----------------------------
# Live Sentiment State (Phase 5.2)
# ----------------------------
st.subheader("🧠 Live Sentiment State")

signals = load_latest_signals()

if signals.empty:
    st.info("No sentiment signals recorded yet.")
else:
    signals["sentiment"] = signals["sentiment"].astype(float)

    def decision_badge(d):
        if d == "buy":
            return "🟢 BUY"
        if d == "sell":
            return "🔴 SELL"
        return "🟡 HOLD"

    display = []
    for _, row in signals.iterrows():
        display.append(
            {
                "Symbol": row["symbol"],
                "Sentiment": round(row["sentiment"], 3),
                "Decision": decision_badge(row["decision"]),
                "Buy ≥": settings.buy_threshold,
                "Sell ≤": settings.sell_threshold,
                "Timestamp": row["timestamp"],
            }
        )

    st.dataframe(display, use_container_width=True)
# ----------------------------
# Cooldown State (Phase 5.3)
# ----------------------------
st.subheader("⏳ Cooldown Status")

cooldowns = load_cooldowns()
now = datetime.utcnow()

if not cooldowns:
    st.success("No active cooldowns — all symbols eligible.")
else:
    rows = []

    for symbol, last_exit in cooldowns.items():
        elapsed = (now - last_exit).total_seconds() / 60
        remaining = max(settings.cooldown_minutes - elapsed, 0)

        rows.append({
            "Symbol": symbol,
            "Last Exit": last_exit.strftime("%Y-%m-%d %H:%M:%S"),
            "Minutes Remaining": round(remaining, 1),
            "Status": "⏳ IN COOLDOWN" if remaining > 0 else "✅ ELIGIBLE",
        })

    st.dataframe(rows, use_container_width=True)
# ----------------------------
# Blocked Reason Visibility (Phase 5.4)
# ----------------------------
st.subheader("🚫 Trade Block Reasons")

blocks = load_block_reasons()

if not blocks:
    st.success("No active trade blocks.")
else:
    rows = []
    for symbol, info in blocks.items():
        rows.append({
            "Symbol": symbol,
            "Blocked Reason": info["reason"],
            "Updated": info["timestamp"],
        })

    st.dataframe(rows, use_container_width=True)
# ----------------------------
# Reconciliation (Phase 5.6.3)
# ----------------------------
st.subheader("🧩 Position Reconciliation")

st.caption(
    "Comparison between positions reconstructed from DB trades "
    "and live broker positions. Read-only safety view."
)

try:
    recon = reconcile_positions(broker)
    recon_ok = bool(recon.get("ok", False))
    recon_summary = dict(recon.get("summary", {}))
    recon_df = pd.DataFrame(recon.get("rows", []))
except Exception as e:
    st.error(f"Failed to load reconciliation data: {e}")
    st.stop()

if recon_df.empty:
    st.success("✅ No open positions detected (DB and broker both empty).")
else:
    # Status summary
    status_counts = recon_df["status"].value_counts().to_dict()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MATCH", status_counts.get("MATCH", 0))
    c2.metric("QTY MISMATCH", status_counts.get("QTY_MISMATCH", 0))
    c3.metric("GHOST (Broker)", status_counts.get("GHOST_BROKER_POSITION", 0))
    c4.metric("GHOST (DB)", status_counts.get("GHOST_DB_POSITION", 0))

    st.divider()

    def highlight(row):
        if row["status"] == "MATCH":
            return ["background-color: #e6fffa"] * len(row)
        if row["status"] == "QTY_MISMATCH":
            return ["background-color: #fff3cd"] * len(row)
        return ["background-color: #f8d7da"] * len(row)

    st.dataframe(
        recon_df.style
        .apply(highlight, axis=1)
        .format(
            {
                "db_qty": "{:.0f}",
                "broker_qty": "{:.0f}",
                "qty_diff": "{:+.0f}",
            }
        ),
        use_container_width=True,
    )

    if recon_df["status"].ne("MATCH").any():
       st.warning(
        "⚠️ Position mismatches detected. "
        "Trading will halt until you unblock after review."
    )
    else:
        st.success("🟢 All positions reconciled successfully.")

    # Mismatch investigation helper
    mismatch_symbols = recon_df.loc[
        recon_df["status"] == "QTY_MISMATCH", "symbol"
    ].astype(str).tolist()
    if mismatch_symbols:
        st.subheader("🔍 Mismatch Investigation")
        sym = st.selectbox("Symbol", mismatch_symbols, key="mismatch_symbol")
        broker_qty = (
            recon_df.loc[recon_df["symbol"] == sym, "broker_qty"].iloc[0]
            if sym in recon_df["symbol"].values
            else None
        )
        db_qty = (
            recon_df.loc[recon_df["symbol"] == sym, "db_qty"].iloc[0]
            if sym in recon_df["symbol"].values
            else None
        )
        st.caption(f"DB qty: {db_qty} | Broker qty: {broker_qty}")
        trades_df = load_df(
            "SELECT timestamp, side, qty, price, order_id FROM trades "
            "WHERE symbol = ? ORDER BY timestamp DESC LIMIT 50",
            (sym,),
        )
        if trades_df.empty:
            st.info("No trades found in DB for this symbol.")
        else:
            st.dataframe(trades_df, use_container_width=True)

    st.subheader("🧯 Controlled Recovery (Phase 5.6.5)")

    halt_state = load_halt_state()
    current_fp = compute_fingerprint(recon_summary)

    if halt_state and halt_state.is_halted:
        if recon_ok:
            st.success("✅ Reconciliation is CLEAN right now. You may unblock trading.")

            note = st.text_input(
                "Unblock note (optional)",
                value="Verified positions; safe to resume.",
                key="unblock_note",
            )

            if st.button(
                "✅ UNBLOCK TRADING (writes unblock_ack.json)",
                key="unblock_trading_btn",
            ):
                write_unblock_ack(current_fp, note=note)
                st.success("Unblock ack written. Restart bot to resume.")
                st.code(f"Fingerprint: {current_fp}")
        else:
            st.warning("⚠️ Reconciliation is NOT clean — unblock is disabled.")
            st.json(recon_summary)
    else:
        st.info("No active halt found.")
        st.code(f"Current fingerprint: {current_fp}")

    if st.button("🧹 Clear Halt State (admin)", key="clear_halt_btn"):
        clear_halt()
        st.success("Halt state cleared.")

    st.subheader("🛠️ Rebuild DB From Broker (admin)")
    st.warning(
        "This will DELETE all trades in the DB and rebuild positions from current broker holdings."
    )
    confirm_rebuild = st.checkbox("I understand — rebuild the DB trades", key="confirm_rebuild_db")
    if st.button("🔁 Rebuild Trades From Broker", key="rebuild_trades_btn", disabled=not confirm_rebuild):
        rebuilt = rebuild_trades_from_broker(broker)
        st.success(f"Rebuilt trades for: {rebuilt}")

    # Live mode helper: DB has positions, broker is empty
    if (
        settings.trading_mode.lower() == "live"
        and recon_summary.get("ghost_db", 0) > 0
        and recon_summary.get("ghost_broker", 0) == 0
    ):
        st.subheader("🧹 Live DB Cleanup Suggestion")
        st.info(
            "Live account has no positions, but the DB shows paper trades. "
            "You can clear and rebuild the DB from live broker positions (empty)."
        )
        confirm_live_clean = st.checkbox(
            "I understand — clear paper DB trades for live account",
            key="confirm_live_clean",
        )
        if st.button(
            "✅ Clear Paper DB (Live)",
            key="clear_paper_db_live_btn",
            disabled=not confirm_live_clean,
        ):
            rebuilt = rebuild_trades_from_broker(broker)
            st.success("DB cleared from live broker (empty).")
