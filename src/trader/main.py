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
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import time

from src.trader.config import settings
from src.trader.brokers.alpaca import AlpacaBroker
from src.trader.risk.risk_manager import RiskManager, RiskLimits
from src.trader.risk.pnl import calculate_daily_pnl

from src.trader.sentiment.finbert_model import FinBertSentimentModel
from src.trader.sentiment.news_fetcher import NewsFetcher
from src.trader.strategy.sentiment_strategy import SentimentStrategy

from src.trader.storage.database import init_db, get_connection
from src.trader.storage.trade_logger import log_trade, log_order_attempt
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
from src.trader.state.auto_heal import (
    clear_db_positions,
    add_broker_positions_to_db,
    sync_db_positions_with_broker,
)
from src.trader.state.symbols import load_symbols
from src.trader.backtest.price_loader import fetch_daily_bars_alpaca


fill_timeout_alerted: set[str] = set()





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


def wait_for_order_fill(
    broker: AlpacaBroker,
    order,
    symbol: str,
    side: str,
    fallback_price: float | None,
) -> dict | None:
    order_id = str(getattr(order, "id", "")) if order is not None else ""
    try:
        result = broker.wait_for_fill(
            order_id=order_id,
            timeout_s=settings.order_fill_timeout_s,
            poll_s=settings.order_fill_poll_s,
        )
    except Exception as e:
        logger.warning(f"Order fill wait failed {symbol} {side} ({order_id}): {e}")
        return None

    if not result or not result.get("filled"):
        status = "unknown"
        o = result.get("order") if result else None
        if o is not None:
            status = str(getattr(o, "status", "unknown"))
        logger.warning(
            f"Order not filled yet; skipping log {symbol} {side} ({order_id}) status={status}"
        )
        try:
            log_order_attempt(
                symbol=symbol,
                side=side,
                qty=0,
                price=fallback_price,
                sentiment=None,
                order_id=order_id or "UNKNOWN",
                status="fill_timeout",
                reason=f"status={status}",
            )
        except Exception:
            pass
        if (
            settings.send_emails
            and settings.send_fill_timeout_email
            and order_id
            and order_id not in fill_timeout_alerted
        ):
            fill_timeout_alerted.add(order_id)
            try:
                send_email(
                    subject=f"⚠️ Order Fill Timeout: {symbol} {side.upper()}",
                    body=(
                        f"Symbol: {symbol}\n"
                        f"Side: {side.upper()}\n"
                        f"Order ID: {order_id}\n"
                        f"Status: {status}\n"
                        f"Timeout: {settings.order_fill_timeout_s}s\n"
                        f"Poll: {settings.order_fill_poll_s}s\n"
                        f"Fallback price: {fallback_price}\n"
                    ),
                    settings=settings,
                )
            except Exception as e:
                logger.error(f"Fill-timeout email failed: {e}")
        return None

    o = result.get("order")
    filled_qty = float(getattr(o, "filled_qty", 0) or 0)
    filled_price = float(getattr(o, "filled_avg_price", 0) or 0)
    if filled_price <= 0 and fallback_price is not None:
        filled_price = float(fallback_price)
    if filled_qty <= 0:
        return None

    return {
        "order_id": order_id,
        "qty": int(filled_qty),
        "price": float(filled_price),
    }


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


def compute_symbol_quality_metrics(lookback_closed_trades: int = 20) -> dict[str, dict]:
    """
    Compute per-symbol realized-trade quality metrics from DB trades using FIFO matching.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT timestamp, symbol, side, qty, price
            FROM trades
            ORDER BY timestamp ASC
            """
        ).fetchall()
    except Exception as e:
        logger.warning(f"Symbol quality metrics unavailable: {e}")
        conn.close()
        return {}
    conn.close()

    fifo: dict[str, list[list[float]]] = {}
    realized_by_symbol: dict[str, list[float]] = {}

    for _, symbol, side, qty, price in rows:
        sym = str(symbol).upper()
        side_l = str(side).lower()
        trade_qty = int(qty)
        trade_px = float(price)

        if sym not in fifo:
            fifo[sym] = []
        if sym not in realized_by_symbol:
            realized_by_symbol[sym] = []

        if side_l == "buy":
            fifo[sym].append([trade_qty, trade_px])
        elif side_l == "sell":
            remaining = trade_qty
            while remaining > 0 and fifo[sym]:
                lot_qty, lot_px = fifo[sym][0]
                matched = min(remaining, int(lot_qty))
                pnl = (trade_px - float(lot_px)) * matched
                realized_by_symbol[sym].append(float(pnl))
                remaining -= matched
                lot_qty = int(lot_qty) - matched
                if lot_qty <= 0:
                    fifo[sym].pop(0)
                else:
                    fifo[sym][0][0] = lot_qty

    if lookback_closed_trades > 0:
        for sym in list(realized_by_symbol.keys()):
            realized_by_symbol[sym] = realized_by_symbol[sym][-lookback_closed_trades:]

    metrics: dict[str, dict] = {}
    for sym, pnls in realized_by_symbol.items():
        nonzero = [p for p in pnls if p != 0]
        if not nonzero:
            continue
        wins = [p for p in nonzero if p > 0]
        losses = [p for p in nonzero if p < 0]
        closed = len(nonzero)
        win_rate = (len(wins) / closed * 100.0) if closed > 0 else 0.0
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(losses) / len(losses)) if losses else 0.0
        expectancy = sum(nonzero) / closed if closed > 0 else 0.0
        metrics[sym] = {
            "closed_trades": int(closed),
            "win_rate_pct": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "expectancy": float(expectancy),
            "total_pnl": float(sum(nonzero)),
        }
    return metrics


def filter_symbols_by_quality(
    symbols: list[str],
    protected_symbols: set[str] | None = None,
) -> list[str]:
    if not settings.symbol_quality_filter_enabled:
        return [s.upper() for s in symbols]

    protected = {s.upper() for s in (protected_symbols or set())}
    min_closed = max(1, int(settings.symbol_quality_min_closed_trades))
    min_expectancy = float(settings.symbol_quality_min_expectancy_usd)
    min_win_rate = float(settings.symbol_quality_min_win_rate_pct)
    lookback_closed = max(0, int(settings.symbol_quality_lookback_closed_trades))

    metrics = compute_symbol_quality_metrics(lookback_closed_trades=lookback_closed)
    kept: list[str] = []

    for symbol in symbols:
        sym = str(symbol).upper()
        m = metrics.get(sym)

        # Never filter out symbols with an open position; exits still need to run.
        if sym in protected:
            kept.append(sym)
            continue

        # Keep symbols with insufficient realized history; gate only when sample is mature.
        if not m or int(m["closed_trades"]) < min_closed:
            kept.append(sym)
            continue

        if float(m["expectancy"]) < min_expectancy or float(m["win_rate_pct"]) < min_win_rate:
            reason = (
                f"Symbol quality filter blocked: closed={m['closed_trades']} "
                f"expectancy={m['expectancy']:.2f} win_rate={m['win_rate_pct']:.1f}%"
            )
            logger.info(f"{sym} | {reason}")
            save_block_reason(sym, reason)
            continue

        kept.append(sym)

    if not kept:
        logger.warning("Symbol quality filter removed all symbols; using configured list.")
        print("[yellow]Symbol quality filter removed all symbols; using configured list.[/yellow]")
        return [s.upper() for s in symbols]

    if len(kept) < len(symbols):
        print(
            f"[cyan]Symbol quality filter active:[/cyan] "
            f"trading {len(kept)}/{len(symbols)} symbols"
        )

    return kept


def get_last_closed_trade_timestamp() -> datetime | None:
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT timestamp
            FROM trades
            WHERE LOWER(side) = 'sell'
            ORDER BY timestamp DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to fetch last closed trade: {e}")
        return None

    if not row or not row[0]:
        return None

    try:
        ts = datetime.fromisoformat(row[0])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts
    except Exception:
        return None


def is_market_open_now() -> bool:
    try:
        tz = ZoneInfo(settings.market_timezone)
    except Exception:
        tz = ZoneInfo("America/New_York")

    now = datetime.now(tz)
    if now.weekday() >= 5:
        return False

    try:
        open_h, open_m = [int(x) for x in settings.market_open_time.split(":")]
        close_h, close_m = [int(x) for x in settings.market_close_time.split(":")]
    except Exception:
        open_h, open_m = 9, 30
        close_h, close_m = 16, 0

    open_t = now.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    close_t = now.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    return open_t <= now <= close_t


def should_run_test_trade() -> bool:
    if not settings.one_share_test_trade:
        return False
    if not settings.test_trade_once_per_day:
        return True

    try:
        tz = ZoneInfo(settings.market_timezone)
    except Exception:
        tz = ZoneInfo("America/New_York")

    today = datetime.now(tz).date().isoformat()
    marker = Path("data/test_trade_last.txt")
    try:
        if marker.exists():
            last = marker.read_text().strip()
            if last == today:
                return False
    except Exception:
        pass
    return True


def mark_test_trade_ran() -> None:
    try:
        tz = ZoneInfo(settings.market_timezone)
    except Exception:
        tz = ZoneInfo("America/New_York")
    today = datetime.now(tz).date().isoformat()
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/test_trade_last.txt").write_text(today + "\n")


def broker_fallback_sell(
    broker: AlpacaBroker,
    symbol: str,
    pre_qty: int,
    sentiment: float | None,
    reason: str,
    price_hint: float | None,
) -> int:
    """
    If an order was placed but no fill confirmation arrived, check broker positions.
    If qty dropped, log a SELL based on broker truth.
    Returns sold qty (0 if none).
    """
    try:
        post_qty = broker.get_position_qty(symbol)
    except Exception:
        return 0

    sold_qty = max(int(pre_qty) - int(post_qty), 0)
    if sold_qty <= 0:
        return 0

    price = price_hint
    if price is None:
        price = get_last_price_safe(broker, symbol, f"broker_fallback_{reason}")

    try:
        log_trade(
            symbol=symbol,
            side="sell",
            qty=int(sold_qty),
            price=float(price or 0.0),
            sentiment=float(sentiment or 0.0),
            order_id=f"BROKER_FALLBACK_{reason}",
        )
    except Exception:
        pass

    try:
        log_order_attempt(
            symbol=symbol,
            side="sell",
            qty=int(sold_qty),
            price=price,
            sentiment=sentiment,
            order_id=f"BROKER_FALLBACK_{reason}",
            status="broker_fallback",
            reason=reason,
        )
    except Exception:
        pass

    try:
        send_email(
            subject=f"📉 Trade Executed (Broker Truth): {symbol} SELL",
            body=(
                f"Symbol: {symbol}\n"
                f"Side: SELL\n"
                f"Qty: {int(sold_qty)}\n"
                f"Price: ${float(price or 0.0):.2f}\n"
                f"Reason: {reason}\n"
                "Note: Fill confirmation missing; logged using broker position change.\n"
            ),
            settings=settings,
        )
    except Exception:
        pass

    return sold_qty


def main():
    # ---- quick visibility of config thresholds ----
    print("THRESHOLDS:", settings.buy_threshold, settings.sell_threshold)
    write_heartbeat("boot")

    # ----- PROOF FILE -----
    Path("TASK_RAN.txt").write_text(f"Task ran at {datetime.now().isoformat()}\n")

    print("[bold cyan]Booting Sentiment Trader[/bold cyan]")
    init_db()

    # ----- STALE CLOSED-TRADES WARNING -----
    if settings.send_emails and settings.send_no_closed_trades_email:
        last_closed = get_last_closed_trade_timestamp()
        now_utc = datetime.now(timezone.utc)
        if last_closed is None:
            try:
                send_email(
                    subject="⚠️ No Closed Trades Found",
                    body=(
                        "No SELL trades found in the database yet.\n"
                        "This can happen if orders never fill, the bot halts early, "
                        "or logging is skipped.\n"
                        f"Time (UTC): {now_utc.isoformat()}\n"
                    ),
                    settings=settings,
                )
            except Exception as e:
                logger.error(f"No-closed-trades email failed: {e}")
        else:
            days_since = (now_utc - last_closed).total_seconds() / 86400.0
            if days_since >= float(settings.no_closed_trades_days_threshold):
                try:
                    send_email(
                        subject="⚠️ Stale Closed Trades",
                        body=(
                            f"Last SELL trade: {last_closed.isoformat()}\n"
                            f"Days since last close: {days_since:.2f}\n"
                            f"Threshold: {settings.no_closed_trades_days_threshold} days\n"
                            f"Time (UTC): {now_utc.isoformat()}\n"
                        ),
                        settings=settings,
                    )
                except Exception as e:
                    logger.error(f"Stale-closed-trades email failed: {e}")

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

    # ----- CANCEL STALE OPEN ORDERS -----
    if settings.cancel_stale_open_orders:
        try:
            canceled = broker.cancel_stale_open_orders(settings.stale_order_minutes)
            if canceled:
                logger.warning(f"Cancelled {canceled} stale open orders.")
        except Exception as e:
            logger.warning(f"Failed to cancel stale orders: {e}")

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
        if (
            settings.trading_mode.lower() != "live"
            and settings.auto_rebuild_on_recon_mismatch
        ):
            try:
                broker_positions = broker.get_positions()
                positions_payload = []
                for p in broker_positions:
                    sym = getattr(p, "symbol", None)
                    if not sym:
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
                sync_db_positions_with_broker(
                    reason="AUTO_REBUILD_ON_RECON_MISMATCH",
                    positions=positions_payload,
                )
                logger.warning("AUTO-REBUILD APPLIED — rechecking reconciliation.")
                print("🩹 AUTO-REBUILD APPLIED — rechecking reconciliation.")
                recon = reconcile_positions(broker)
                recon_ok = bool(recon.get("ok", False))
                recon_summary = dict(recon.get("summary", {}))
            except Exception as e:
                logger.error(f"AUTO-REBUILD FAILED: {e}")
                print(f"[red]AUTO-REBUILD FAILED[/red]: {e}")

        if recon_ok:
            logger.warning("AUTO-REBUILD SUCCESS — reconciliation OK, continuing.")
        else:
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
                    target_symbols = sorted(
                        set([s.upper() for s in ghost_db_symbols + ghost_broker_symbols])
                    )
                    positions_payload = []
                    for sym in target_symbols:
                        p = broker_map.get(sym.upper())
                        if p:
                            positions_payload.append(
                                {
                                    "symbol": sym,
                                    "qty": int(getattr(p, "qty", 0)),
                                    "avg_entry_price": float(
                                        getattr(p, "avg_entry_price", 0.0) or 0.0
                                    ),
                                }
                            )
                        else:
                            positions_payload.append(
                                {
                                    "symbol": sym,
                                    "qty": 0,
                                    "avg_entry_price": 0.0,
                                }
                            )
                    if target_symbols:
                        sync_db_positions_with_broker(
                            reason=decision["reason"],
                            positions=positions_payload,
                            symbols=target_symbols,
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
        elif (
            settings.trading_mode.lower() != "live"
            and settings.auto_unblock_on_clean_recon
            and recon_ok
        ):
            print("✅ AUTO-UNBLOCK — reconciliation is clean (paper mode).")
            logger.info("CONTROLLED RECOVERY — auto-unblock (paper mode, clean recon).")
            clear_halt()
            logger.info("CONTROLLED RECOVERY — halt state cleared (auto).")
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
        positions = []

    # ----- BUYING POWER CHECK (LIVE ONLY) -----
    if not paper and buying_power <= 0:
        print("[bold red]NO BUYING POWER — SKIPPING TRADES[/bold red]")
        logger.error("NO BUYING POWER — fund live account before trading")
        return

    # ----- MARKET HOURS GUARD -----
    if settings.market_hours_only and not is_market_open_now():
        print("[yellow]Market closed — skipping trading actions[/yellow]")
        logger.warning("Market closed — skipping trading actions")
        return

    # ----- ONE-SHARE TEST TRADE (PAPER ONLY) -----
    if paper and not read_only and should_run_test_trade():
        test_symbol = settings.test_trade_symbol.upper().strip() or "AAPL"
        if not settings.market_hours_only or is_market_open_now():
            try:
                print(f"[yellow]TEST TRADE → BUY 1 {test_symbol}[/yellow]")
                order = broker.place_market_order(test_symbol, "buy", 1)
                try:
                    log_order_attempt(
                        symbol=test_symbol,
                        side="buy",
                        qty=1,
                        price=None,
                        sentiment=None,
                        order_id=str(getattr(order, "id", "TEST_BUY")),
                        status="submitted",
                        reason="test_trade_buy",
                    )
                except Exception:
                    pass

                fill = wait_for_order_fill(
                    broker=broker,
                    order=order,
                    symbol=test_symbol,
                    side="buy",
                    fallback_price=None,
                )
                if fill:
                    log_trade(
                        symbol=test_symbol,
                        side="buy",
                        qty=int(fill["qty"]),
                        price=float(fill["price"]),
                        sentiment=0.0,
                        order_id=str(fill["order_id"] or "TEST_BUY"),
                    )

                    if settings.test_trade_round_trip:
                        print(f"[yellow]TEST TRADE → SELL 1 {test_symbol}[/yellow]")
                        order2 = broker.place_market_order(test_symbol, "sell", 1)
                        try:
                            log_order_attempt(
                                symbol=test_symbol,
                                side="sell",
                                qty=1,
                                price=None,
                                sentiment=None,
                                order_id=str(getattr(order2, "id", "TEST_SELL")),
                                status="submitted",
                                reason="test_trade_sell",
                            )
                        except Exception:
                            pass

                        fill2 = wait_for_order_fill(
                            broker=broker,
                            order=order2,
                            symbol=test_symbol,
                            side="sell",
                            fallback_price=None,
                        )
                        if not fill2:
                            broker_fallback_sell(
                                broker=broker,
                                symbol=test_symbol,
                                pre_qty=1,
                                sentiment=0.0,
                                reason="TEST_TRADE",
                                price_hint=None,
                            )
                        else:
                            log_trade(
                                symbol=test_symbol,
                                side="sell",
                                qty=int(fill2["qty"]),
                                price=float(fill2["price"]),
                                sentiment=0.0,
                                order_id=str(fill2["order_id"] or "TEST_SELL"),
                            )

                mark_test_trade_ran()
            except Exception as e:
                logger.error(f"Test trade failed: {e}")

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
    held_symbols = {
        str(getattr(p, "symbol", "")).upper()
        for p in positions
        if getattr(p, "symbol", "")
    }
    symbols = sorted(set(symbols) | held_symbols)
    symbols = filter_symbols_by_quality(symbols, protected_symbols=held_symbols)
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

                pre_qty = current_qty
                pre_qty = current_qty
                order = broker.place_market_order(symbol, "sell", current_qty)
                try:
                    log_order_attempt(
                        symbol=symbol,
                        side="sell",
                        qty=current_qty,
                        price=None,
                        sentiment=sentiment_score,
                        order_id=str(getattr(order, "id", "TIME_EXIT")),
                        status="submitted",
                        reason="time_exit",
                    )
                except Exception:
                    pass

                exit_price = get_last_price_safe(broker, symbol, "time_exit")
                fill = wait_for_order_fill(
                    broker=broker,
                    order=order,
                    symbol=symbol,
                    side="sell",
                    fallback_price=exit_price,
                )
                if not fill:
                    sold_qty = broker_fallback_sell(
                        broker=broker,
                        symbol=symbol,
                        pre_qty=pre_qty,
                        sentiment=sentiment_score,
                        reason="TIME_EXIT",
                        price_hint=exit_price,
                    )
                    if sold_qty <= 0:
                        time.sleep(2)
                        continue
                    exit_price = float(exit_price or 0.0)
                    fill = {
                        "qty": sold_qty,
                        "price": exit_price,
                        "order_id": "BROKER_FALLBACK_TIME_EXIT",
                    }
                exit_price = float(fill["price"])
                log_trade(
                    symbol=symbol,
                    side="sell",
                    qty=int(fill["qty"]),
                    price=exit_price,
                    sentiment=sentiment_score,
                    order_id=str(fill["order_id"] or "TIME_EXIT"),
                )

                last_trade_time[symbol] = datetime.utcnow()
                cooldowns[symbol] = datetime.utcnow()
                save_cooldowns(cooldowns)

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

                pre_qty = current_qty
                order = broker.place_market_order(symbol, "sell", current_qty)
                try:
                    log_order_attempt(
                        symbol=symbol,
                        side="sell",
                        qty=current_qty,
                        price=None,
                        sentiment=sentiment_score,
                        order_id=str(getattr(order, "id", "EXIT")),
                        status="submitted",
                        reason="signal_exit",
                    )
                except Exception:
                    pass

                # 🔁 RESET SENTIMENT BIAS (PHASE 3.1)
                sentiment_streak[symbol] = {"buy": 0, "sell": 0}
                save_streaks(sentiment_streak)

                exit_price = get_last_price_safe(broker, symbol, "exit")
                fill = wait_for_order_fill(
                    broker=broker,
                    order=order,
                    symbol=symbol,
                    side="sell",
                    fallback_price=exit_price,
                )
                if not fill:
                    sold_qty = broker_fallback_sell(
                        broker=broker,
                        symbol=symbol,
                        pre_qty=pre_qty,
                        sentiment=sentiment_score,
                        reason="SIGNAL_EXIT",
                        price_hint=exit_price,
                    )
                    if sold_qty <= 0:
                        time.sleep(2)
                        continue
                    exit_price = float(exit_price or 0.0)
                    fill = {
                        "qty": sold_qty,
                        "price": exit_price,
                        "order_id": "BROKER_FALLBACK_SIGNAL_EXIT",
                    }
                exit_price = float(fill["price"])
                save_block_reason(symbol, "ELIGIBLE")

                log_trade(
                    symbol=symbol,
                    side="sell",
                    qty=int(fill["qty"]),
                    price=exit_price,
                    sentiment=sentiment_score,
                    order_id=str(fill["order_id"] or "EXIT"),
                )

                exit_time = datetime.utcnow()
                last_trade_time[symbol] = exit_time
                cooldowns[symbol] = exit_time
                save_cooldowns(cooldowns)

                send_email(
                    subject=f"📉 Trade Executed: {symbol} SELL",
                    body=(
                        f"Symbol: {symbol}\n"
                        f"Side: SELL\n"
                        f"Qty: {int(fill['qty'])}\n"
                        f"Price: ${exit_price:.2f}\n"
                        f"Sentiment: {sentiment_score:.3f}\n"
                        f"Order ID: {getattr(order, 'id', 'EXIT')}"
                    ),
                    settings=settings,
                )

                state["trades_today"] += 1

                print(f"[red]SELL submitted[/red]: {getattr(order, 'id', 'EXIT')}")
                logger.info(
                    f"TRADE symbol={symbol} side=sell qty={int(fill['qty'])} "
                    f"sentiment={sentiment_score:.3f} price~{exit_price:.2f}"
                )
            else:
                print("No position to sell — skipping")

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

                    pre_qty = current_qty
                    order = broker.place_market_order(symbol, "sell", current_qty)
                    try:
                        log_order_attempt(
                            symbol=symbol,
                            side="sell",
                            qty=current_qty,
                            price=None,
                            sentiment=sentiment_score,
                            order_id=str(getattr(order, "id", "TP_SL")),
                            status="submitted",
                            reason=reason.lower(),
                        )
                    except Exception:
                        pass

                    exit_price = get_last_price_safe(broker, symbol, "tp_sl_exit")
                    fill = wait_for_order_fill(
                        broker=broker,
                        order=order,
                        symbol=symbol,
                        side="sell",
                        fallback_price=exit_price if exit_price is not None else float(last_px),
                    )
                    if not fill:
                        sold_qty = broker_fallback_sell(
                            broker=broker,
                            symbol=symbol,
                            pre_qty=pre_qty,
                            sentiment=sentiment_score,
                            reason=reason,
                            price_hint=exit_price if exit_price is not None else float(last_px),
                        )
                        if sold_qty <= 0:
                            time.sleep(2)
                            continue
                        exit_price = float((exit_price if exit_price is not None else float(last_px)) or 0.0)
                        fill = {
                            "qty": sold_qty,
                            "price": exit_price,
                            "order_id": f"BROKER_FALLBACK_{reason}",
                        }
                    exit_price = float(fill["price"])
                    log_trade(
                        symbol=symbol,
                        side="sell",
                        qty=int(fill["qty"]),
                        price=exit_price,
                        sentiment=sentiment_score,
                        order_id=str(fill["order_id"] or "TP_SL"),
                    )

                    last_trade_time[symbol] = datetime.utcnow()
                    cooldowns[symbol] = datetime.utcnow()
                    save_cooldowns(cooldowns)

                    send_email(
                        subject=f"📉 Trade Executed: {symbol} SELL",
                        body=(
                            f"Symbol: {symbol}\n"
                            f"Side: SELL\n"
                            f"Qty: {int(fill['qty'])}\n"
                            f"Price: ${exit_price:.2f}\n"
                            f"Reason: {reason}\n"
                            f"Order ID: {fill['order_id'] or getattr(order, 'id', 'TP_SL')}"
                        ),
                        settings=settings,
                    )

                    state["trades_today"] += 1
                    print(f"[red]SELL submitted[/red]: {getattr(order, 'id', 'TP_SL')}")
                    logger.info(
                        f"TRADE symbol={symbol} side=sell qty={int(fill['qty'])} "
                        f"price~{exit_price:.2f} reason={reason.lower()}"
                    )

                    sentiment_streak[symbol] = {"buy": 0, "sell": 0}
                    save_streaks(sentiment_streak)
                    entry_times.pop(symbol, None)
                    peak_prices.pop(symbol, None)
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
            try:
                log_order_attempt(
                    symbol=symbol,
                    side="buy",
                    qty=qty,
                    price=price,
                    sentiment=sentiment_score,
                    order_id=str(getattr(order, "id", "ENTRY")),
                    status="submitted",
                    reason="signal_entry",
                )
            except Exception:
                pass
            save_block_reason(symbol, "Eligible")
            fill = wait_for_order_fill(
                broker=broker,
                order=order,
                symbol=symbol,
                side="buy",
                fallback_price=price,
            )
            if not fill:
                time.sleep(2)
                continue
            log_trade(
                symbol=symbol,
                side="buy",
                qty=int(fill["qty"]),
                price=float(fill["price"]),
                sentiment=sentiment_score,
                order_id=str(fill["order_id"] or "ENTRY"),
            )

            last_trade_time[symbol] = datetime.utcnow()

            send_email(
                subject=f"📈 Trade Executed: {symbol} BUY",
                body=(
                    f"Symbol: {symbol}\n"
                    f"Side: BUY\n"
                    f"Qty: {int(fill['qty'])}\n"
                    f"Price: ${float(fill['price']):.2f}\n"
                    f"Sentiment: {sentiment_score:.3f}\n"
                    f"Order ID: {fill['order_id'] or getattr(order, 'id', 'ENTRY')}"
                ),
                settings=settings,
            )

            state["trades_today"] += 1

            print(f"[green]BUY submitted[/green]: {getattr(order, 'id', 'ENTRY')}")
            logger.info(
                f"TRADE symbol={symbol} side=buy qty={int(fill['qty'])} "
                f"sentiment={sentiment_score:.3f} price~{float(fill['price']):.2f}"
            )

            if scale_in:
                cooldowns[symbol] = datetime.utcnow()
                save_cooldowns(cooldowns)
                logger.info(f"SCALE-IN cooldown set for {symbol}")

            time.sleep(2)
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
