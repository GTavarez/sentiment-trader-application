import pandas as pd


def run_backtest_with_prices(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    buy_threshold: float,
    sell_threshold: float,
    starting_cash: float = 100_000.0,
    trade_size_usd: float = 500.0,
    slippage_bps: float = 2.0,
    fee_per_trade: float = 0.50,
):
    # --- Guards ---
    if signals.empty or prices.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "starting_cash": starting_cash,
            "ending_equity": starting_cash,
            "net_pnl": 0.0,
            "trades": 0,
        }

    sig = signals.copy()
    px = prices.copy()

    sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True)
    px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True)

    sig = sig.sort_values("timestamp")
    px = px.sort_values(["symbol", "timestamp"])

    cash = float(starting_cash)
    positions = {}   # symbol -> shares
    trades = []
    equity_curve = []

    slip = slippage_bps / 10_000.0

    # Helper: last price
    def last_price(symbol, ts):
        s = px[(px["symbol"] == symbol) & (px["timestamp"] <= ts)]
        return float(s.iloc[-1]["close"]) if not s.empty else None

    # === MAIN LOOP ===
    for _, row in sig.iterrows():
        ts = row["timestamp"]
        symbol = row["symbol"]
        sentiment = float(row["sentiment"])
        held = positions.get(symbol, 0)

        price = last_price(symbol, ts)
        if price is None:
            continue

        side = None
        if sentiment >= buy_threshold:
            side = "buy"
        """ elif sentiment <= sell_threshold and held > 0:
            side = "sell" """
        

        # === EXECUTE TRADE ===
        if side == "buy":
            qty = max(1, int(trade_size_usd // price))
            cost = qty * price * (1 + slip) + fee_per_trade

            if cash >= cost:
                cash -= cost
                positions[symbol] = held + qty
                trades.append({
                    "timestamp": ts,
                    "symbol": symbol,
                    "side": "buy",
                    "qty": qty,
                    "price": price,
                    "sentiment": sentiment,
                    "cash_after": cash,
                })

        elif side == "sell":
            proceeds = held * price * (1 - slip) - fee_per_trade
            cash += proceeds
            positions[symbol] = 0
            trades.append({
                "timestamp": ts,
                "symbol": symbol,
                "side": "sell",
                "qty": held,
                "price": price,
                "sentiment": sentiment,
                "cash_after": cash,
            })

        # === MARK TO MARKET (EVERY STEP) ===
        equity = cash
        for sym, shares in positions.items():
             last_px = prices[
                 (prices["symbol"] == sym) &
                 (prices["timestamp"] <= ts)
    ]

    if not last_px.empty:
        equity += shares * float(last_px.iloc[-1]["close"])

        equity_curve.append({
            "timestamp": ts,
            "equity": equity,
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    ending_equity = equity_df["equity"].iloc[-1]

    summary = {
        "starting_cash": starting_cash,
        "ending_equity": ending_equity,
        "net_pnl": ending_equity - starting_cash,
        "trades": len(trades_df),
        "buys": int((trades_df["side"] == "buy").sum()) if not trades_df.empty else 0,
        "sells": int((trades_df["side"] == "sell").sum()) if not trades_df.empty else 0,
        "positions_end": positions,
    }

    return trades_df, equity_df, summary

