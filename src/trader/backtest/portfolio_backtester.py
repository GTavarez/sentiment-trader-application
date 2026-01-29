import pandas as pd
from typing import Dict, Tuple
from datetime import timedelta


def run_portfolio_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
    starting_cash: float = 100_000.0,
    trade_risk_pct: float = 0.05,
) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Portfolio backtest with shared capital.

    - No merge_asof (prevents sorting errors)
    - Marks equity every step
    - One position per symbol
    - Time-based + sentiment exits
    """

    # -----------------------------
    # Guard clause
    # -----------------------------
    if signals.empty or prices.empty:
        return pd.DataFrame(), {}, pd.DataFrame()

    # -----------------------------
    # Normalize inputs
    # -----------------------------
    sig = signals.copy()
    px = prices.copy()

    sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True)
    px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True)

    sig["symbol"] = sig["symbol"].astype(str).str.upper()
    px["symbol"] = px["symbol"].astype(str).str.upper()

    sig = sig.sort_values("timestamp").reset_index(drop=True)
    px = px.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------
    # State
    # -----------------------------
    cash = starting_cash
    positions: Dict[str, int] = {}
    entry_time: Dict[str, pd.Timestamp] = {}
    daily_trade_count: Dict[pd.Timestamp.date, int] = {}

    equity_curve = []
    trades = []

    MAX_HOLD = timedelta(hours=4)  # ⬅️ time-based exit (Option C)
    MAX_POSITION_PCT = 0.20  # 20% max exposure per symbol
    MAX_TOTAL_EXPOSURE_PCT = 0.60  # 60% invested max
    MAX_TRADES_PER_DAY = 5


    # -----------------------------
    # Helper: latest price lookup
    # -----------------------------
    def latest_price(symbol: str, ts: pd.Timestamp):
        rows = px[(px["symbol"] == symbol) & (px["timestamp"] <= ts)]
        if rows.empty:
            return None
        return float(rows.iloc[-1]["close"])

    # -----------------------------
    # Main loop (signal-driven)
    # -----------------------------
    for _, row in sig.iterrows():
        ts = row["timestamp"]
        symbol = row["symbol"]
        sentiment = float(row["sentiment"])

        price = latest_price(symbol, ts)
        if price is None:
            continue

        held = positions.get(symbol, 0)
        trade_day = ts.date()
        daily_trade_count.setdefault(trade_day, 0)

        # -----------------------------
        # EXIT: time-based
        # -----------------------------
        if held > 0 and symbol in entry_time:
            if ts - entry_time[symbol] >= MAX_HOLD:
                cash += held * price
                positions[symbol] = 0
                entry_time.pop(symbol, None)

                trades.append(
                    (ts, symbol, "sell", held, price, sentiment)
                )
                daily_trade_count[trade_day] += 1

                held = 0  # reset

        # -----------------------------
        # EXIT: sentiment-based
        # -----------------------------
        if sentiment <= sell_threshold and held > 0:
            cash += held * price
            positions[symbol] = 0
            entry_time.pop(symbol, None)

            trades.append(
                (ts, symbol, "sell", held, price, sentiment)
            )
            daily_trade_count[trade_day] += 1
            held = 0  # reset
        # EXIT: weakening sentiment (trend decay)
        if held > 0 and sentiment < buy_threshold:
            cash += held * price
            positions[symbol] = 0
            entry_time.pop(symbol, None)

            trades.append(
               (ts, symbol, "sell", held, price, sentiment)
    )


        # -----------------------------
        # ENTRY
        # -----------------------------
        if (
            sentiment >= buy_threshold
            and held == 0
            and daily_trade_count[trade_day] < MAX_TRADES_PER_DAY
        ):
              # EXIT: neutral sentiment reset
            if -0.15 < sentiment < 0.15 and held > 0:
                cash += held * price
                positions[symbol] = 0
                entry_time.pop(symbol, None)

                trades.append(
                   (ts, symbol, "sell", held, price, sentiment)
    )
            held = 0

            equity = cash + sum(
                 positions.get(sym, 0) * latest_price(sym, ts)
                 for sym in positions
                 if latest_price(sym, ts) is not None)

              # Exposure checks
            current_exposure = sum(
                  positions.get(sym, 0) * latest_price(sym, ts)
                  for sym in positions
                  if latest_price(sym, ts) is not None
            )

            if current_exposure >= equity * MAX_TOTAL_EXPOSURE_PCT:
                  pass  # portfolio full
            else:
                 max_position_value = equity * MAX_POSITION_PCT
                 risk_cash = min(equity * trade_risk_pct, max_position_value)

                 qty = max(1, int(risk_cash // price))
                 cost = qty * price
                 if cash >= cost:
                    cash -= cost
                    positions[symbol] = qty
                    entry_time[symbol] = ts

                    trades.append((ts, symbol, "buy", qty, price, sentiment))
                    daily_trade_count[trade_day] += 1
        # -----------------------------
        # MARK TO MARKET (EVERY STEP)
        # -----------------------------
        equity = cash + sum(
            positions.get(sym, 0) * latest_price(sym, ts)
            for sym in positions
            if latest_price(sym, ts) is not None
        )

        equity_curve.append(
            {
                "timestamp": ts,
                "equity": equity,
                "cash": cash,
            }
        )

    # -----------------------------
    # Outputs
    # -----------------------------
    equity_df = pd.DataFrame(equity_curve)

    trades_df = pd.DataFrame(
        trades,
        columns=["timestamp", "symbol", "side", "qty", "price", "sentiment"],
    )

    summary = {
        "starting_cash": starting_cash,
        "ending_equity": float(equity_df["equity"].iloc[-1])
        if not equity_df.empty
        else starting_cash,
        "net_pnl": float(
            equity_df["equity"].iloc[-1] - starting_cash
        )
        if not equity_df.empty
        else 0.0,
        "trades": len(trades_df),
    }

    return equity_df, summary, trades_df
