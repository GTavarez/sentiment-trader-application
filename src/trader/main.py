from rich import print
from loguru import logger
from pathlib import Path
from datetime import datetime
from datetime import timedelta

from src.trader.config import settings
from src.trader.brokers.alpaca import AlpacaBroker
from src.trader.risk.risk_manager import RiskManager, RiskLimits
from src.trader.risk.pnl import calculate_daily_pnl

from src.trader.sentiment.finbert_model import FinBertSentimentModel
from src.trader.sentiment.news_fetcher import NewsFetcher
from src.trader.strategy.sentiment_strategy import SentimentStrategy

from src.trader.storage.database import init_db
from src.trader.storage.signal_logger import log_signal
from src.trader.storage.trade_logger import log_trade
from src.trader.Notifications.emailer import send_email



def in_cooldown(symbol: str, cooldowns: dict, cooldown_minutes: int) -> bool:
    last_exit = cooldowns.get(symbol)
    if not last_exit:
        return False
    return datetime.utcnow() < last_exit + timedelta(minutes=cooldown_minutes)
print("THRESHOLDS:", settings.buy_threshold, settings.sell_threshold)
def main():
    # ----- PROOF FILE -----
    Path("TASK_RAN.txt").write_text(
        f"Task ran at {datetime.now().isoformat()}\n"
    )

    print("[bold cyan]Booting Sentiment Trader[/bold cyan]")
    init_db()
    last_trade_time = {}
    COOLDOWN_MINUTES = 30

    # ----- MODE -----
    trading_mode = settings.trading_mode.lower()
    paper = trading_mode != "live"

    print("[green]Trading mode: PAPER[/green]" if paper else "[bold red]LIVE TRADING ENABLED[/bold red]")

    # ----- BROKER -----
    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        paper=paper,
    )

    try:
        account = broker.get_account()
        print(f"Account equity: ${account.equity}")
    except RuntimeError as e:
        print("[bold red]BROKER UNAVAILABLE — SKIPPING RUN[/bold red]")
        logger.error(e)
        return
    positions = broker.get_positions()
    print("ALPACA POSITIONS AT START:")
    for p in positions:
         print(p.symbol, p.qty)

    # ----- SENTIMENT STACK -----
    sentiment_model = FinBertSentimentModel()
    news_fetcher = NewsFetcher(settings.news_api_key)

    strategy = SentimentStrategy(
        buy_threshold=settings.buy_threshold,
        sell_threshold=settings.sell_threshold,
    )
    assert strategy.buy_threshold > strategy.sell_threshold, \
    f"Invalid thresholds: buy={strategy.buy_threshold}, sell={strategy.sell_threshold}"

    # ----- DAILY PNL / KILL SWITCH -----
    daily_pnl = calculate_daily_pnl(broker)
    print(f"Daily PnL: ${daily_pnl:.2f}")

    if daily_pnl <= -settings.daily_loss_limit_usd:
        print("[bold red]KILL SWITCH ACTIVATED — DAILY LOSS LIMIT HIT[/bold red]")
        return

    # ----- RISK -----
    limits = RiskLimits(
        max_trades_per_day=settings.max_trades_per_day,
        max_position_usd=settings.max_position_usd,
        daily_loss_limit_usd=settings.daily_loss_limit_usd,
    )
    risk = RiskManager(limits)

    state = {
        "trades_today": 0,
        "pnl_today_usd": daily_pnl,
    }
    cooldowns = {}
    # ----- MAIN LOOP -----
    symbols = settings.symbol_list
    
    for symbol in symbols:
        print(f"\n[bold]=== Processing {symbol} ===[/bold]")

        headlines = news_fetcher.fetch(symbol, limit=10)
        sentiment_score = sentiment_model.score_texts(headlines)

        decision_pack = strategy.decide(
        sentiment_score=sentiment_score,
        timestamps=[datetime.now()] * len(headlines),    )

        decision = decision_pack["decision"]
        current_qty = broker.get_position_qty(symbol)
        logger.info(
            f"SENTIMENT DEBUG | {symbol} | score={sentiment_score:.3f} | "
            f"buy={settings.buy_threshold} sell={settings.sell_threshold}")

        print(f"Sentiment score: {sentiment_score:.3f}")
        print(f"Decision: {decision.upper()} | Current Qty: {current_qty}")
        broker.cancel_open_orders(symbol)
        # ⛔ COOLDOWN BLOCK
        if decision == "buy" and in_cooldown(symbol, cooldowns, settings.cooldown_minutes):
             print(f"⏳ {symbol} in cooldown — skipping BUY")
             continue

    # ----------------------------------
    # 🚨 EXIT FIRST (NO FLIPS)
    # ----------------------------------
        if decision == "sell" and sentiment_score <= strategy.sell_threshold:
           if current_qty > 0:
              print(f"[red]EXIT → SELL {current_qty} {symbol}[/red]")
              broker.place_market_order(symbol, "sell", current_qty)
              log_trade(
                 symbol=symbol,
                 side="sell",
                 qty=current_qty,
                 price=broker.get_last_price(symbol),
                 sentiment=sentiment_score,
                 order_id="EXIT",
            )
           else:
            print("No position to sell — skipping")

           cooldowns[symbol] = datetime.utcnow()
           continue
           
        if decision == "hold":
            print("[yellow]No trade — HOLD[/yellow]")
            continue
    # ----------------------------------
    # BUY ONLY IF FLAT
    # ----------------------------------
        if decision == "buy" and sentiment_score >= strategy.buy_threshold:
            if current_qty > 0:
                 print("Already in position — skipping BUY")
                 continue

            price = broker.get_last_price(symbol)
            qty = max(1, int(settings.max_position_usd // price))

            print(f"[green]ENTRY → BUY {qty} {symbol}[/green]")

            order = broker.place_market_order(symbol, "buy", qty)

            log_trade(
               symbol=symbol,
               side="buy",
               qty=qty,
               price=price,
               sentiment=sentiment_score,
               order_id=str(order.id),
        )

        send_email(
            subject=f"📈 Trade Executed: {symbol} BUY",
            body=(
                f"Symbol: {symbol}\n"
                f"Side: BUY\n"
                f"Qty: {qty}\n"
                f"Price: ${price:.2f}\n"
                f"Sentiment: {sentiment_score:.3f}\n"
                f"Order ID: {order.id}"
            ),
            settings=settings,
        )

        state["trades_today"] += 1

        print(f"[green]BUY submitted[/green]: {order.id}")
        logger.info(
            f"TRADE symbol={symbol} side=buy qty={qty} "
            f"sentiment={sentiment_score:.3f} price~{price:.2f}"
        )
        now = datetime.utcnow()

        if symbol in last_trade_time:
           delta = now - last_trade_time[symbol]
        if delta.total_seconds() < COOLDOWN_MINUTES * 60:
            print("Cooldown active — skipping")
        continue

if __name__ == "__main__":
    main()
