# ---- enforce UTF‑8 stdout/stderr for Windows Task Scheduler ----
import sys
import io
import os
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8", errors="replace")
# ----------------------------------------------------------------

from rich import print
from loguru import logger
from pathlib import Path
from datetime import datetime, timedelta
import time

from src.trader.config import settings
from src.trader.brokers.alpaca import AlpacaBroker
from src.trader.risk.risk_manager import RiskManager, RiskLimits
from src.trader.risk.pnl import calculate_daily_pnl

from src.trader.sentiment.finbert_model import FinBertSentimentModel
from src.trader.sentiment.news_fetcher import NewsFetcher
from src.trader.strategy.sentiment_strategy import SentimentStrategy

from src.trader.storage.database import init_db
from src.trader.storage.trade_logger import log_trade
from src.trader.Notifications.emailer import send_email
from src.trader.state.streaks import load_streaks, save_streaks
from src.trader.state.cooldowns import load_cooldowns, save_cooldowns
from src.trader.state.blocks import save_block_reason
from src.trader.state.reconciliation import reconcile_positions
from src.trader.state.halt_state import (
    load_halt_state,
    is_unblocked_for,
    compute_fingerprint,
    write_halt,
    clear_halt,
)
from src.trader.state.recovery import can_auto_heal, auto_heal_action
from src.trader.state.auto_heal import clear_db_positions, add_broker_positions_to_db
from src.trader.state.symbols import load_symbols
from src.trader.backtest.price_loader import fetch_daily_bars_alpaca





def in_cooldown(symbol: str, cooldowns: dict, cooldown_minutes: int) -> bool:
    last_exit = cooldowns.get(symbol)
    if not last_exit:
        return False
    return datetime.utcnow() < last_exit + timedelta(minutes=cooldown_minutes)


def safe_cancel_open_orders(broker: AlpacaBroker, symbol: str) -> None:
    """
    Your broker has cancel_open_orders(symbol) in your logs.
    This wrapper prevents crashes if it ever isn't available.
    """
    fn = getattr(broker, "cancel_open_orders", None)
    if callable(fn):
        fn(symbol)


def write_heartbeat(status: str = "running") -> None:
    try:
        Path("data").mkdir(parents=True, exist_ok=True)
        Path("data/heartbeat.txt").write_text(
            f"{datetime.now().isoformat()} | {status}\n"
        )
    except Exception:
        pass


def get_last_price_safe(
    broker: AlpacaBroker, symbol: str, context: str = ""
) -> float | None:
    try:
        return float(broker.get_last_price(symbol))
    except Exception as e:
        ctx = f" ({context})" if context else ""
        logger.error(f"Price fetch failed for {symbol}{ctx}: {e}")
        save_block_reason(symbol, f"Price fetch failed{ctx}: {e}")
        return None


def build_price_indicators(symbols: list[str], api_key: str, secret_key: str) -> dict:
    if not (settings.trend_filter_enabled or settings.volatility_filter_enabled):
        return {}

    lookback = max(settings.trend_sma_days, settings.volatility_lookback_days, 2)
    try:
        prices = fetch_daily_bars_alpaca(
            api_key=api_key,
            secret_key=secret_key,
            symbols=symbols,
            days=lookback + 5,
        )
    except Exception as e:
        logger.warning(f"Price fetch failed; skipping filters: {e}")
        return {}

    if prices.empty:
        return {}

    prices["symbol"] = prices["symbol"].astype(str).str.upper()
    indicators = {}

    for sym in symbols:
        sdf = prices[prices["symbol"] == sym.upper()].sort_values("timestamp")
        if sdf.empty:
            continue
        closes = sdf["close"].astype(float)
        last_close = float(closes.iloc[-1])

        sma = None
        if len(closes) >= settings.trend_sma_days:
            sma = float(closes.tail(settings.trend_sma_days).mean())

        vol = None
        if len(closes) >= settings.volatility_lookback_days + 1:
            returns = closes.pct_change().dropna()
            vol = float(returns.tail(settings.volatility_lookback_days).std())

        indicators[sym.upper()] = {
            "last_close": last_close,
            "sma": sma,
            "vol": vol,
        }

    return indicators


def main():
    # ---- quick visibility of config thresholds ----
    print("THRESHOLDS:", settings.buy_threshold, settings.sell_threshold)
    write_heartbeat("boot")

    # ----- PROOF FILE -----
    Path("TASK_RAN.txt").write_text(f"Task ran at {datetime.now().isoformat()}\n")

    print("[bold cyan]Booting Sentiment Trader[/bold cyan]")
    init_db()

    # ----- MODE -----
    trading_mode = settings.trading_mode.lower()
    paper = trading_mode != "live"
    if paper:
        print("[green]Trading mode: PAPER[/green]")
    else:
        print("[bold red]LIVE TRADING ENABLED[/bold red]")
        logger.warning("LIVE TRADING ENABLED")
        if not settings.live_trading_confirm:
            print("[bold red]LIVE TRADING NOT CONFIRMED — EXITING[/bold red]")
            logger.error("LIVE TRADING NOT CONFIRMED — set LIVE_TRADING_CONFIRM=true to proceed")
            return
        if settings.live_trading_confirm_code.strip().upper() != "I_UNDERSTAND":
            print("[bold red]LIVE TRADING CONFIRM CODE INVALID — EXITING[/bold red]")
            logger.error("LIVE TRADING CONFIRM CODE INVALID — set LIVE_TRADING_CONFIRM_CODE=I_UNDERSTAND")
            return

    read_only = bool(settings.read_only)
    if read_only:
        print("[yellow]READ-ONLY MODE — no orders will be placed[/yellow]")
        logger.warning("READ-ONLY MODE — no orders will be placed")

    # ----- BROKER -----
    api_key = settings.alpaca_api_key
    secret_key = settings.alpaca_secret_key
    if paper and settings.alpaca_api_key_paper and settings.alpaca_secret_key_paper:
        api_key = settings.alpaca_api_key_paper
        secret_key = settings.alpaca_secret_key_paper
    if not paper and settings.alpaca_api_key_live and settings.alpaca_secret_key_live:
        api_key = settings.alpaca_api_key_live
        secret_key = settings.alpaca_secret_key_live

    if not api_key or not secret_key:
        print("[bold red]ALPACA KEYS MISSING — CHECK .env[/bold red]")
        logger.error("ALPACA KEYS MISSING — set *_PAPER or *_LIVE keys in .env")
        return

    broker = AlpacaBroker(
        api_key=api_key,
        secret_key=secret_key,
        paper=paper,
        last_price_max_retries=settings.last_price_max_retries,
        last_price_retry_base_s=settings.last_price_retry_base_s,
        last_price_retry_max_s=settings.last_price_retry_max_s,
    )
    try:
        key_prefix = api_key[:4]
        print(f"[dim]Alpaca key prefix: {key_prefix}**** | mode={trading_mode}[/dim]")
    except Exception:
        pass

    # ----- SAFE BROKER CHECK -----
    try:
        account = broker.get_account()
        print(f"Account equity: ${account.equity}")
        try:
            buying_power = float(getattr(account, "buying_power", 0.0) or 0.0)
        except Exception:
            buying_power = 0.0
    except RuntimeError as e:
        msg = str(e)
        if "401" in msg or "unauthorized" in msg.lower():
            print("[bold red]BROKER UNAUTHORIZED — CHECK LIVE VS PAPER KEYS[/bold red]")
            logger.error("BROKER UNAUTHORIZED — check Alpaca keys for live/paper mode")
        else:
            print("[bold red]BROKER UNAVAILABLE — SKIPPING RUN[/bold red]")
            logger.error(e)
        return

    # ----- STARTUP EMAIL TEST -----
    if settings.send_emails and settings.send_startup_email_test:
        try:
            send_email(
                subject="🧪 Startup Email Test",
                body=(
                    "This is a startup email test from Sentiment Trader.\n"
                    f"Mode: {settings.trading_mode}\n"
                    f"Time: {datetime.utcnow().isoformat()}Z\n"
                ),
                settings=settings,
            )
        except Exception as e:
            logger.error(f"Startup email test failed: {e}")
        # =========================
    # PHASE 5.6.5 — CONTROLLED RECOVERY GATE
    # =========================
    try:
        recon = reconcile_positions(broker)
        recon_ok = bool(recon.get("ok", False))
        recon_summary = dict(recon.get("summary", {}))
    except Exception as e:
        # If reconciliation itself errors, treat as unsafe and halt.
        recon_ok = False
        recon_summary = {"error": str(e)}
    # If reconciliation failed, attempt safe auto-heal (paper only).
    if not recon_ok:
        if can_auto_heal(
            trading_mode=settings.trading_mode,
            summary=recon_summary,
        ):
            decision = auto_heal_action(recon_summary)

            if decision["action"] == "CLEAR_DB_POSITIONS":
                ghost_symbols = [
                    r.get("symbol")
                    for r in recon.get("rows", [])
                    if r.get("status") == "GHOST_DB_POSITION"
                ]
                ghost_symbols = [s for s in ghost_symbols if s]
                clear_db_positions(reason=decision["reason"], symbols=ghost_symbols)
                logger.warning("AUTO-HEAL APPLIED", decision)
                print("🩹 AUTO-HEAL APPLIED — restart bot to continue")
                return
            if decision["action"] == "REBUILD_DB_FROM_BROKER":
                broker_positions = broker.get_positions()
                broker_map = {
                    p.symbol.upper(): p for p in broker_positions if hasattr(p, "symbol")
                }
                ghost_symbols = [
                    r.get("symbol")
                    for r in recon.get("rows", [])
                    if r.get("status") == "GHOST_BROKER_POSITION"
                ]
                ghost_symbols = [s for s in ghost_symbols if s]
                positions_payload = []
                for sym in ghost_symbols:
                    p = broker_map.get(sym.upper())
                    if not p:
                        continue
                    positions_payload.append(
                        {
                            "symbol": sym,
                            "qty": int(getattr(p, "qty", 0)),
                            "avg_entry_price": float(
                                getattr(p, "avg_entry_price", 0.0) or 0.0
                            ),
                        }
                    )
                add_broker_positions_to_db(
                    reason=decision["reason"],
                    positions=positions_payload,
                )
                logger.warning("AUTO-HEAL APPLIED", decision)
                print("🩹 AUTO-HEAL APPLIED — restart bot to continue")
                return
            if decision["action"] == "SYNC_DB_WITH_BROKER":
                ghost_db_symbols = [
                    r.get("symbol")
                    for r in recon.get("rows", [])
                    if r.get("status") == "GHOST_DB_POSITION"
                ]
                ghost_db_symbols = [s for s in ghost_db_symbols if s]
                if ghost_db_symbols:
                    clear_db_positions(reason=decision["reason"], symbols=ghost_db_symbols)

                broker_positions = broker.get_positions()
                broker_map = {
                    p.symbol.upper(): p for p in broker_positions if hasattr(p, "symbol")
                }
                ghost_broker_symbols = [
                    r.get("symbol")
                    for r in recon.get("rows", [])
                    if r.get("status") == "GHOST_BROKER_POSITION"
                ]
                ghost_broker_symbols = [s for s in ghost_broker_symbols if s]
                positions_payload = []
                for sym in ghost_broker_symbols:
                    p = broker_map.get(sym.upper())
                    if not p:
                        continue
                    positions_payload.append(
                        {
                            "symbol": sym,
                            "qty": int(getattr(p, "qty", 0)),
                            "avg_entry_price": float(
                                getattr(p, "avg_entry_price", 0.0) or 0.0
                            ),
                        }
                    )
                if positions_payload:
                    add_broker_positions_to_db(
                        reason=decision["reason"],
                        positions=positions_payload,
                    )

                logger.warning("AUTO-HEAL APPLIED", decision)
                print("🩹 AUTO-HEAL APPLIED — restart bot to continue")
                return
        # Any mismatch should halt until operator unblocks.
        write_halt(reason="RECON_MISMATCH", details=recon_summary)
        print("🚨 RECONCILIATION MISMATCH — TRADING HALTED")
        logger.error(f"RECONCILIATION MISMATCH — halted: {recon_summary}")
        return
    # Compute the fingerprint for current recon result
    current_fp = compute_fingerprint(recon_summary)
    existing_halt = load_halt_state()
    # If a previous halt exists, require explicit unblock
    if existing_halt and existing_halt.is_halted:
        print(f"🧩 Existing halt detected (fingerprint: {existing_halt.fingerprint})")
        if is_unblocked_for(current_fp):
            print(f"✅ CONTROLLED RECOVERY — fingerprint {current_fp} unblocked, trading may resume.")
            logger.info("CONTROLLED RECOVERY — operator unblock confirmed, trading may resume.")
            # Clear halt state so it does not linger after a successful unblock
            clear_halt()
            logger.info("CONTROLLED RECOVERY — halt state cleared.")
        else:
            print("🚨 TRADING STILL HALTED — waiting for operator UNBLOCK in Streamlit")
            print(f"Current fingerprint: {current_fp}")
            logger.error("CONTROLLED RECOVERY — STILL HALTED (needs operator unblock)")
            return
    else:
        logger.info("CONTROLLED RECOVERY — no active halt, continuing trading.")

    # Print positions at start
    try:
        positions = broker.get_positions()
        print("ALPACA POSITIONS AT START:")
        for p in positions:
            print(p.symbol, p.qty)
    except Exception as e:
        logger.warning(f"Could not fetch positions at start: {e}")

    # ----- BUYING POWER CHECK (LIVE ONLY) -----
    if not paper and buying_power <= 0:
        print("[bold red]NO BUYING POWER — SKIPPING TRADES[/bold red]")
        logger.error("NO BUYING POWER — fund live account before trading")
        return

    # ----- SENTIMENT STACK -----
    sentiment_model = FinBertSentimentModel()
    news_fetcher = NewsFetcher(settings.news_api_key)

    strategy = SentimentStrategy(
        buy_threshold=settings.buy_threshold,
        sell_threshold=settings.sell_threshold,
    )

    # Make sure buy > sell
    assert strategy.buy_threshold > strategy.sell_threshold, (
        f"Invalid thresholds: buy={strategy.buy_threshold}, sell={strategy.sell_threshold}"
    )

    # ----- DAILY PNL / KILL SWITCH -----
    daily_pnl = calculate_daily_pnl(broker)
    print(f"Daily PnL: ${daily_pnl:.2f}")

    if daily_pnl <= -settings.daily_loss_limit_usd:
        save_block_reason("GLOBAL", "Daily loss limit reached")
        write_halt(
            reason="DAILY_LOSS_LIMIT",
            details={
                "daily_pnl": float(daily_pnl),
                "limit": float(settings.daily_loss_limit_usd),
            },
        )
        print("[bold red]KILL SWITCH ACTIVATED — DAILY LOSS LIMIT HIT[/bold red]")
        return

    # ----- RISK MANAGER -----
    limits = RiskLimits(
        max_trades_per_day=settings.max_trades_per_day,
        max_position_usd=settings.max_position_usd,
        daily_loss_limit_usd=settings.daily_loss_limit_usd,
    )
    risk = RiskManager(limits)

    state = {"trades_today": 0, "pnl_today_usd": daily_pnl}

    # ----- COOLDOWNS + STREAKS -----
    cooldowns = load_cooldowns()        # symbol -> datetime of last exit
    last_trade_time = {}  # symbol -> datetime of last trade (entry or exit)
    sentiment_streak = load_streaks()  # symbol -> {"buy": int, "sell": int}

    COOLDOWN_PERIOD = timedelta(minutes=30)
    REQUIRED_CONFIRMATIONS = settings.min_signal_cycles  # streak confirmations (Option A)
    MAX_HOLD_TIME = timedelta(hours=4)
    entry_times = {}  # symbol -> datetime
     

    # ----- MAIN LOOP -----
    symbols = load_symbols(settings.symbol_list)
    max_symbols = int(settings.max_symbols_per_run)
    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    price_indicators = build_price_indicators(symbols, api_key, secret_key)
    tp_pct = float(settings.take_profit_pct)
    sl_pct = float(settings.stop_loss_pct)
    trailing_enabled = bool(settings.trailing_stop_enabled)
    trailing_pct = float(settings.trailing_stop_pct)
    peak_prices = {}

    for symbol in symbols:
        write_heartbeat(f"loop:{symbol}")
        print(f"\n[bold]=== Processing {symbol} ===[/bold]")
        logger.info(f"ENTER LOOP | {symbol}")

        # Fetch + score (THIS MUST BE INSIDE THE SYMBOL LOOP)
        headlines = news_fetcher.fetch(symbol, limit=10)
        sentiment_score = sentiment_model.score_texts(headlines)

        # Current position
        current_qty = broker.get_position_qty(symbol)

        logger.info(
            f"SENTIMENT DEBUG | {symbol} | score={sentiment_score:.3f} | "
            f"buy={settings.buy_threshold} sell={settings.sell_threshold}"
        )

        # ---- streak init ----
        if symbol not in sentiment_streak:
            sentiment_streak[symbol] = {"buy": 0, "sell": 0}

        # ---- update streak counters ----
        if sentiment_score >= strategy.buy_threshold:
            sentiment_streak[symbol]["buy"] += 1
            sentiment_streak[symbol]["sell"] = 0
        elif sentiment_score <= strategy.sell_threshold:
            sentiment_streak[symbol]["sell"] += 1
            sentiment_streak[symbol]["buy"] = 0
        else:
            sentiment_streak[symbol]["buy"] = 0
            sentiment_streak[symbol]["sell"] = 0

        logger.info(
            f"SENTIMENT STREAK | {symbol} | "
            f"buy={sentiment_streak[symbol]['buy']} "
            f"sell={sentiment_streak[symbol]['sell']}"
        )

        # ✅ Persist streaks after updating
        save_streaks(sentiment_streak)

        # ---- decide using streak confirmations ----
        if sentiment_streak[symbol]["buy"] >= REQUIRED_CONFIRMATIONS:
            decision = "buy"
        elif sentiment_streak[symbol]["sell"] >= REQUIRED_CONFIRMATIONS:
            decision = "sell"
        else:
            decision = "hold"

        print(f"Sentiment score: {sentiment_score:.3f}")
        print(f"Decision: {decision.upper()} | Current Qty: {current_qty}")

        # ---- trailing stop tracking (only for open positions) ----
        if current_qty > 0 and trailing_enabled:
            last_px_for_trail = get_last_price_safe(broker, symbol, "trailing")
            if last_px_for_trail is not None:
                prev_peak = peak_prices.get(symbol, last_px_for_trail)
                peak_prices[symbol] = max(prev_peak, last_px_for_trail)

        # ---- strategy filters (BUY only; SELL optional) ----
        filter_sell = bool(settings.apply_filters_to_sell)
        if decision == "buy" or (decision == "sell" and filter_sell):
            ind = price_indicators.get(symbol.upper())
            if settings.trend_filter_enabled and ind and ind.get("sma") is not None:
                if ind["last_close"] < ind["sma"]:
                    save_block_reason(symbol, "Trend filter: price below SMA")
                    print(f"📉 Trend filter blocked {decision.upper()} — {symbol} below SMA")
                    time.sleep(2)
                    continue

            if settings.volatility_filter_enabled and ind and ind.get("vol") is not None:
                if ind["vol"] > settings.max_daily_volatility_pct:
                    save_block_reason(
                        symbol,
                        f"Volatility filter: {ind['vol']:.4f} > {settings.max_daily_volatility_pct:.4f}",
                    )
                    print(f"🌪️ Volatility filter blocked {decision.upper()} — {symbol} too volatile")
                    time.sleep(2)
                    continue

        # ---- cooldown checks ----
        per_symbol_cd = {
            k.upper(): int(v) for k, v in settings.per_symbol_cooldown_minutes.items()
        }
        cooldown_minutes = per_symbol_cd.get(symbol.upper(), settings.cooldown_minutes)
        if decision == "buy" and in_cooldown(symbol, cooldowns, cooldown_minutes):
            print(f"⏳ {symbol} in cooldown — skipping BUY")
            time.sleep(2)
            continue

        now = datetime.utcnow()
        if symbol in last_trade_time:
            entry_times[symbol] = datetime.utcnow()

            elapsed = now - last_trade_time[symbol]
            if elapsed < COOLDOWN_PERIOD:
                save_block_reason(symbol, "Cooldown period active")
                print(
                    f"⏳ Cooldown active for {symbol} "
                    f"({int(elapsed.total_seconds() // 60)}m elapsed)"
                )
                time.sleep(2)
                # 🔄 reset sentiment streak after BUY
                sentiment_streak[symbol] = {"buy": 0, "sell": 0}
                save_streaks(sentiment_streak)
                save_block_reason(symbol, "Cooldown window not elapsed")

                continue
        # ⏱ TIME-BASED EXIT
        if current_qty > 0 and symbol in entry_times:
            held_for = datetime.utcnow() - entry_times[symbol]
            if held_for >= MAX_HOLD_TIME:
                 print(f"[red]TIME EXIT → SELL {current_qty} {symbol}[/red]")

            if read_only:
                logger.info(f"READ-ONLY — would TIME EXIT sell {current_qty} {symbol}")
                time.sleep(2)
                continue

            order = broker.place_market_order(symbol, "sell", current_qty)
            last_trade_time[symbol] = datetime.utcnow()
            cooldowns[symbol] = datetime.utcnow()

            exit_price = get_last_price_safe(broker, symbol, "time_exit")
            if exit_price is None:
                exit_price = 0.0
            log_trade(
                symbol=symbol,
                side="sell",
                qty=current_qty,
                price=exit_price,
                sentiment=sentiment_score,
                order_id=str(getattr(order, "id", "TIME_EXIT")),
        )

        # 🔄 reset state
            sentiment_streak[symbol] = {"buy": 0, "sell": 0}
            entry_times.pop(symbol, None)
            save_streaks(sentiment_streak)

            continue

        # ----------------------------------
        # EXIT FIRST (NO FLIPS)
        # ----------------------------------
        if decision == "sell":
            if current_qty > 0:
                safe_cancel_open_orders(broker, symbol)
                print(f"[red]EXIT → SELL {current_qty} {symbol}[/red]")

                if read_only:
                    logger.info(f"READ-ONLY — would EXIT sell {current_qty} {symbol}")
                    time.sleep(2)
                    continue

                order = broker.place_market_order(symbol, "sell", current_qty)

                exit_time = datetime.utcnow()
                last_trade_time[symbol] = exit_time
                cooldowns[symbol] = exit_time
                save_cooldowns(cooldowns)


        # 🔁 RESET SENTIMENT BIAS (PHASE 3.1)
                sentiment_streak[symbol] = {"buy": 0, "sell": 0}
                save_streaks(sentiment_streak)

                exit_price = get_last_price_safe(broker, symbol, "exit")
                if exit_price is None:
                    exit_price = 0.0
                save_block_reason(symbol, "ELIGIBLE")

                log_trade(
                   symbol=symbol,
                   side="sell",
                   qty=current_qty,
                   price=exit_price,
                   sentiment=sentiment_score,
                   order_id=str(getattr(order, "id", "EXIT")),
        )

                send_email(
                   subject=f"📉 Trade Executed: {symbol} SELL",
                   body=(
                      f"Symbol: {symbol}\n"
                      f"Side: SELL\n"
                      f"Qty: {current_qty}\n"
                      f"Price: ${exit_price:.2f}\n"
                      f"Sentiment: {sentiment_score:.3f}\n"
                      f"Order ID: {getattr(order, 'id', 'EXIT')}"
                      ),
                     settings=settings,
                     )

                state["trades_today"] += 1

                print(f"[red]SELL submitted[/red]: {getattr(order, 'id', 'EXIT')}")
                logger.info(
                       f"TRADE symbol={symbol} side=sell qty={current_qty} "
                       f"sentiment={sentiment_score:.3f} price~{exit_price:.2f}"
                         )
            else:
                print("No position to sell — skipping")

            continue

        
        # ----------------------------------
        # HOLD
        # ----------------------------------
        if decision == "hold":
            print("[yellow]No trade — HOLD[/yellow]")
            time.sleep(2)
            continue

        # ----------------------------------
        # BUY ONLY IF FLAT
        # ----------------------------------
        if decision == "buy":
            scale_in = current_qty > 0 and settings.allow_scale_in
            if current_qty > 0 and not settings.allow_scale_in:
                save_block_reason(symbol, "Already in position")
                print("Already in position — skipping BUY")
                time.sleep(2)
                continue
            elif scale_in:
                save_block_reason(symbol, "Scale-in allowed")
                print("Scale-in enabled — evaluating additional BUY")

            price = get_last_price_safe(broker, symbol, "entry")
            if price is None:
                print("[red]Price unavailable — skipping BUY[/red]")
                time.sleep(2)
                continue
            per_symbol_pos = {
                k.upper(): float(v) for k, v in settings.max_position_usd_by_symbol.items()
            }
            max_pos_usd = per_symbol_pos.get(symbol.upper(), settings.max_position_usd)
            qty = max(1, int(max_pos_usd // price))

            if scale_in:
                per_symbol_caps = {
                    k.upper(): v for k, v in settings.max_symbol_exposure_by_symbol.items()
                }
                symbol_cap = per_symbol_caps.get(symbol.upper(), settings.max_symbol_exposure_usd)
                current_notional = current_qty * price
                proposed_notional = qty * price
                total_notional = current_notional + proposed_notional
                if total_notional > symbol_cap:
                    save_block_reason(
                        symbol,
                        f"Scale-in cap exceeded: ${total_notional:.2f} > "
                        f"${symbol_cap:.2f}",
                    )
                    print(
                        f"[red]Scale-in blocked[/red]: "
                        f"total ${total_notional:.2f} > "
                        f"cap ${symbol_cap:.2f}"
                    )
                    if settings.send_scale_in_block_email:
                        send_email(
                            subject=f"⚠️ Scale-in blocked: {symbol}",
                            body=(
                                f"Symbol: {symbol}\n"
                                f"Current qty: {current_qty}\n"
                                f"Price: ${price:.2f}\n"
                                f"Current notional: ${current_notional:.2f}\n"
                                f"Proposed notional: ${proposed_notional:.2f}\n"
                                f"Total notional: ${total_notional:.2f}\n"
                                f"Cap: ${symbol_cap:.2f}\n"
                            ),
                            settings=settings,
                        )
                    time.sleep(2)
                    continue

            proposed = {
                "symbol": symbol,
                "side": "buy",
                "qty": qty,
                "price": price,
                "notional_usd": qty * price,
            }
            ok, reason = risk.allow_trade(state, proposed)
            if not ok:
                save_block_reason(symbol, f"Risk blocked: {reason}")
                risk.log_block(proposed, reason)
                print(f"[red]Trade blocked[/red]: {reason}")
                time.sleep(2)
                continue

            safe_cancel_open_orders(broker, symbol)
            print(f"[green]ENTRY → BUY {qty} {symbol}[/green]")

            if read_only:
                logger.info(f"READ-ONLY — would BUY {qty} {symbol}")
                time.sleep(2)
                continue

            order = broker.place_market_order(symbol, "buy", qty)
            last_trade_time[symbol] = datetime.utcnow()
            save_block_reason(symbol, "Eligible")
            log_trade(
                symbol=symbol,
                side="buy",
                qty=qty,
                price=price,
                sentiment=sentiment_score,
                order_id=str(getattr(order, "id", "ENTRY")),
            )

            send_email(
                subject=f"📈 Trade Executed: {symbol} BUY",
                body=(
                    f"Symbol: {symbol}\n"
                    f"Side: BUY\n"
                    f"Qty: {qty}\n"
                    f"Price: ${price:.2f}\n"
                    f"Sentiment: {sentiment_score:.3f}\n"
                    f"Order ID: {getattr(order, 'id', 'ENTRY')}"
                ),
                settings=settings,
            )

            state["trades_today"] += 1

            print(f"[green]BUY submitted[/green]: {getattr(order, 'id', 'ENTRY')}")
            logger.info(
                f"TRADE symbol={symbol} side=buy qty={qty} "
                f"sentiment={sentiment_score:.3f} price~{price:.2f}"
            )

            if scale_in:
                cooldowns[symbol] = datetime.utcnow()
                save_cooldowns(cooldowns)
                logger.info(f"SCALE-IN cooldown set for {symbol}")

            time.sleep(2)
            continue

        # 🎯 TAKE-PROFIT / STOP-LOSS / TRAILING STOP EXIT
        if current_qty > 0:
            last_px = get_last_price_safe(broker, symbol, "tp_sl_check")
            try:
                avg_entry = float(getattr(broker.get_position(symbol), "avg_entry_price", 0.0))
            except Exception:
                avg_entry = 0.0

            if last_px is not None and avg_entry > 0:
                change_pct = (last_px - avg_entry) / avg_entry
                trailing_hit = False
                if trailing_enabled:
                    peak = peak_prices.get(symbol, last_px)
                    if last_px <= peak * (1 - trailing_pct):
                        trailing_hit = True

                if change_pct >= tp_pct:
                    reason = "TAKE_PROFIT"
                    print(f"[green]TAKE PROFIT → SELL {current_qty} {symbol}[/green]")
                elif change_pct <= -sl_pct:
                    reason = "STOP_LOSS"
                    print(f"[red]STOP LOSS → SELL {current_qty} {symbol}[/red]")
                elif trailing_hit:
                    reason = "TRAILING_STOP"
                    print(f"[red]TRAILING STOP → SELL {current_qty} {symbol}[/red]")
                else:
                    reason = ""

                if reason:
                    if read_only:
                        logger.info(f"READ-ONLY — would {reason} sell {current_qty} {symbol}")
                        time.sleep(2)
                        continue

                    order = broker.place_market_order(symbol, "sell", current_qty)
                    last_trade_time[symbol] = datetime.utcnow()
                    cooldowns[symbol] = datetime.utcnow()
                    save_cooldowns(cooldowns)

                    exit_price = get_last_price_safe(broker, symbol, "tp_sl_exit")
                    if exit_price is None:
                        exit_price = float(last_px)
                    log_trade(
                        symbol=symbol,
                        side="sell",
                        qty=current_qty,
                        price=exit_price,
                        sentiment=sentiment_score,
                        order_id=str(getattr(order, "id", "TP_SL")),
                    )

                    send_email(
                        subject=f"📉 Trade Executed: {symbol} SELL",
                        body=(
                            f"Symbol: {symbol}\n"
                            f"Side: SELL\n"
                            f"Qty: {current_qty}\n"
                            f"Price: ${exit_price:.2f}\n"
                            f"Reason: {reason}\n"
                            f"Order ID: {getattr(order, 'id', 'TP_SL')}"
                        ),
                        settings=settings,
                    )

                    state["trades_today"] += 1
                    print(f"[red]SELL submitted[/red]: {getattr(order, 'id', 'TP_SL')}")
                    logger.info(
                        f"TRADE symbol={symbol} side=sell qty={current_qty} "
                        f"price~{exit_price:.2f} reason={reason.lower()}"
                    )

                    sentiment_streak[symbol] = {"buy": 0, "sell": 0}
                    save_streaks(sentiment_streak)
                    entry_times.pop(symbol, None)
                    peak_prices.pop(symbol, None)
                    continue

    # ----- DAILY SUMMARY EMAIL -----
    if settings.send_daily_summary_email:
        try:
            positions = broker.get_positions()
            if positions:
                positions_lines = [
                    f"{p.symbol} qty={p.qty} mv=${float(p.market_value):,.2f}"
                    for p in positions
                ]
                positions_block = "\n".join(positions_lines)
            else:
                positions_block = "No open positions"

            summary_body = (
                f"Mode: {settings.trading_mode.upper()}\n"
                f"Date (UTC): {datetime.utcnow().strftime('%Y-%m-%d')}\n"
                f"Trades today: {state['trades_today']}\n"
                f"Daily PnL: ${daily_pnl:.2f}\n"
                f"Open positions:\n{positions_block}\n"
            )

            send_email(
                subject="📬 Sentiment Trader — Daily Summary",
                body=summary_body,
                settings=settings,
            )
        except Exception as e:
            logger.error(f"Failed to send daily summary email: {e}")

    # ----- DAILY HEALTH REPORT EMAIL -----
    if settings.send_daily_health_email:
        try:
            health_body = (
                f"Mode: {settings.trading_mode.upper()}\n"
                f"Read-only: {read_only}\n"
                f"Date (UTC): {datetime.utcnow().strftime('%Y-%m-%d')}\n"
                f"Trades today: {state['trades_today']}\n"
                f"Daily PnL: ${daily_pnl:.2f}\n"
                f"Symbols: {', '.join(symbols)}\n"
                f"Filters: trend={settings.trend_filter_enabled} "
                f"vol={settings.volatility_filter_enabled} "
                f"tp={settings.take_profit_pct:.2%} "
                f"sl={settings.stop_loss_pct:.2%} "
                f"trail={'on' if settings.trailing_stop_enabled else 'off'}\n"
            )
            send_email(
                subject="🩺 Sentiment Trader — Daily Health Report",
                body=health_body,
                settings=settings,
            )
        except Exception as e:
            logger.error(f"Failed to send daily health email: {e}")


if __name__ == "__main__":
    main()
