from rich import print
from loguru import logger
from pathlib import Path
from datetime import datetime

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


def main():
    # ----- HARD PROOF FILE -----
    proof = Path("TASK_RAN.txt")
    proof.write_text(f"Task ran at {datetime.now().isoformat()}\n")

    print("[bold cyan]Booting Sentiment Trader[/bold cyan]")
    init_db()

    # ----- TRADING MODE -----
    trading_mode = settings.trading_mode.lower()
    paper = trading_mode != "live"

    if trading_mode == "live":
        print("[bold red]LIVE TRADING ENABLED[/bold red]")
        logger.warning("LIVE TRADING ENABLED")
    else:
        print("[green]Trading mode: PAPER[/green]")

    # ----- BROKER -----
    broker = AlpacaBroker(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        paper=paper,
    )

    # ----- SAFE BROKER CHECK -----
    try:
        account = broker.get_account()
        print(f"Account equity: ${account.equity}")
    except RuntimeError as e:
        print("[bold red]BROKER UNAVAILABLE — SKIPPING RUN[/bold red]")
        logger.error(e)
        return

    # ----- SENTIMENT MODEL (LOAD EARLY / CACHE WARM) -----
    sentiment_model = FinBertSentimentModel()

    # ----- REAL DAILY PNL -----
    daily_pnl = calculate_daily_pnl(broker)
    print(f"Daily PnL: ${daily_pnl:.2f}")

    if daily_pnl <= -settings.daily_loss_limit_usd:
        print("[bold red]KILL SWITCH ACTIVATED — DAILY LOSS LIMIT HIT[/bold red]")
        return

    # ----- STATE -----
    state = {
        "trades_today": 0,
        "pnl_today_usd": daily_pnl,
    }

    # ----- RISK MANAGER -----
    limits = RiskLimits(
        max_trades_per_day=settings.max_trades_per_day,
        max_position_usd=settings.max_position_usd,
        daily_loss_limit_usd=settings.daily_loss_limit_usd,
    )
    risk = RiskManager(limits)

    # ----- SENTIMENT PIPELINE -----
    symbol = "AAPL"

    news_fetcher = NewsFetcher(settings.news_api_key)
    strategy = SentimentStrategy(
        buy_threshold=settings.buy_threshold,
        sell_threshold=settings.sell_threshold,
    )

    headlines = news_fetcher.fetch(symbol, limit=10)

    print(f"Fetched {len(headlines)} headlines")
    for h in headlines:
        logger.debug(f"HEADLINE: {h}")

    sentiment_score = sentiment_model.score_texts(headlines)
    decision = strategy.decide(sentiment_score)

    print(f"Sentiment score for {symbol}: {sentiment_score:.3f}")
    print(f"Strategy decision: {decision.upper()}")

    # ----- LOG SIGNAL -----
    log_signal(
        symbol=symbol,
        sentiment=sentiment_score,
        decision=decision,
    )

    if decision == "hold":
        print("[yellow]No trade — HOLD[/yellow]")
        return

    # ----- ORDER PROPOSAL -----
    price = broker.get_last_price(symbol)
    qty = max(1, int(settings.max_position_usd // price))

    # 🔒 LIVE SAFETY: FORCE TINY SIZE
    if trading_mode == "live":
        qty = min(qty, 1)

    notional = qty * price

    proposed = {
        "symbol": symbol,
        "side": decision,
        "qty": qty,
        "price": price,
        "notional_usd": notional,
    }

    ok, reason = risk.allow_trade(state, proposed)
    if not ok:
        risk.log_block(proposed, reason)
        print(f"[red]Trade blocked[/red]: {reason}")
        return

    # ----- EXECUTION -----
    order = broker.place_market_order(
        symbol=symbol,
        side=decision,
        qty=qty,
    )

    log_trade(
        symbol=symbol,
        side=decision,
        qty=qty,
        price=price,
        sentiment=sentiment_score,
        order_id=order.id,
    )
    send_email(
    subject=f"📈 Trade Executed: {symbol} {decision.upper()}",
    body=(
        f"Trade executed successfully.\n\n"
        f"Symbol: {symbol}\n"
        f"Side: {decision.upper()}\n"
        f"Quantity: {qty}\n"
        f"Price: ${price:.2f}\n"
        f"Sentiment Score: {sentiment_score:.3f}\n"
        f"Order ID: {order.id}"
    ),
    settings=settings,
)
    state["trades_today"] += 1

    print(
        f"[green]{'LIVE' if trading_mode == 'live' else 'PAPER'} "
        f"order submitted[/green]: {order.id}"
    )

    logger.info(
        f"TRADE symbol={symbol} side={decision} qty={qty} "
        f"sentiment={sentiment_score:.3f} price~{price:.2f} "
        f"mode={trading_mode}"
    )


if __name__ == "__main__":
    main()
