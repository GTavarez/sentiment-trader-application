from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from loguru import logger

DB_PATH = Path("data/trader.db")


def _load_net_qty_map(cur: sqlite3.Cursor) -> dict[str, int]:
    cur.execute(
        """
        SELECT UPPER(symbol) AS symbol,
               COALESCE(SUM(
                   CASE
                       WHEN LOWER(side) = 'buy' THEN CAST(qty AS INTEGER)
                       ELSE -CAST(qty AS INTEGER)
                   END
               ), 0) AS net_qty
        FROM trades
        GROUP BY UPPER(symbol)
        """
    )
    out: dict[str, int] = {}
    for sym, qty in cur.fetchall():
        if not sym:
            continue
        out[str(sym).upper()] = int(qty or 0)
    return out


def _last_price(cur: sqlite3.Cursor, symbol: str, fallback: float = 0.0) -> float:
    cur.execute(
        """
        SELECT price
        FROM trades
        WHERE UPPER(symbol) = ? AND price IS NOT NULL
        ORDER BY timestamp DESC, id DESC
        LIMIT 1
        """,
        (symbol.upper(),),
    )
    row = cur.fetchone()
    if not row:
        return float(fallback)
    try:
        return float(row[0] or fallback)
    except Exception:
        return float(fallback)


def _insert_trade(
    cur: sqlite3.Cursor,
    *,
    symbol: str,
    side: str,
    qty: int,
    price: float,
    reason: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    cur.execute(
        """
        INSERT INTO trades (timestamp, symbol, side, qty, price, sentiment, order_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (now, symbol.upper(), side.lower(), int(qty), float(price), 0.0, reason),
    )

def clear_db_positions(reason: str, symbols: list[str] | None = None) -> None:
    """
    Flattens DB open-position state with synthetic SELL trades (PAPER ONLY).
    This preserves trade history and closed-trade stats.
    Used when DB shows ghost positions but broker is clean.
    """
    if not DB_PATH.exists():
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    qty_map = _load_net_qty_map(cur)

    flattened: list[str] = []
    if symbols:
        target_symbols = [s.upper() for s in symbols if str(s).strip()]
    else:
        target_symbols = list(qty_map.keys())

    for sym in target_symbols:
        net_qty = int(qty_map.get(sym, 0))
        if net_qty <= 0:
            continue
        px = _last_price(cur, sym, fallback=0.0)
        _insert_trade(
            cur,
            symbol=sym,
            side="sell",
            qty=net_qty,
            price=px,
            reason=f"AUTO_HEAL_FLATTEN_{reason}",
        )
        flattened.append(f"{sym}:{net_qty}")

    conn.commit()
    conn.close()

    logger.warning(
        f"AUTO-HEAL APPLIED — DB positions flattened ({reason}) "
        f"{'symbols=' + ','.join(flattened) if flattened else 'symbols=none'}"
    )


def add_broker_positions_to_db(
    reason: str,
    positions: list[dict],
) -> None:
    """
    Insert synthetic BUY delta trades for broker-held positions (PAPER ONLY).
    Only tops up missing quantity to match broker; does not wipe history.
    Each position dict: {symbol, qty, avg_entry_price}
    """
    if not DB_PATH.exists():
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    qty_map = _load_net_qty_map(cur)

    inserted = []
    for p in positions:
        symbol = str(p.get("symbol", "")).upper()
        target_qty = int(p.get("qty", 0))
        price = float(p.get("avg_entry_price", 0.0) or 0.0)
        if not symbol or target_qty <= 0:
            continue
        current_qty = int(qty_map.get(symbol, 0))
        missing_qty = max(0, target_qty - current_qty)
        if missing_qty <= 0:
            continue
        if price <= 0:
            price = _last_price(cur, symbol, fallback=0.0)
        _insert_trade(
            cur,
            symbol=symbol,
            side="buy",
            qty=missing_qty,
            price=price,
            reason=f"AUTO_HEAL_BROKER_TOPUP_{reason}",
        )
        inserted.append(f"{symbol}:{missing_qty}")
        qty_map[symbol] = current_qty + missing_qty

    conn.commit()
    conn.close()

    logger.warning(
        f"AUTO-HEAL APPLIED — DB positions topped up from broker ({reason}) "
        f"{'symbols=' + ','.join(inserted) if inserted else 'symbols=none'}"
    )


def sync_db_positions_with_broker(
    reason: str,
    positions: list[dict],
    symbols: list[str] | None = None,
) -> None:
    """
    Non-destructive reconciliation:
    - Inserts synthetic BUY/SELL delta trades so DB net qty matches broker qty.
    - Preserves historical trades and closed-trade metrics.
    """
    if not DB_PATH.exists():
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    qty_map = _load_net_qty_map(cur)

    broker_map: dict[str, dict] = {}
    for p in positions:
        sym = str(p.get("symbol", "")).upper()
        if not sym:
            continue
        broker_map[sym] = {
            "qty": int(p.get("qty", 0) or 0),
            "avg_entry_price": float(p.get("avg_entry_price", 0.0) or 0.0),
        }

    if symbols:
        target_symbols = {str(s).upper() for s in symbols if str(s).strip()}
    else:
        target_symbols = set(qty_map.keys()) | set(broker_map.keys())

    changes: list[str] = []
    for sym in sorted(target_symbols):
        db_qty = int(qty_map.get(sym, 0))
        broker_qty = int(broker_map.get(sym, {}).get("qty", 0))
        delta = broker_qty - db_qty
        if delta == 0:
            continue

        broker_px = float(broker_map.get(sym, {}).get("avg_entry_price", 0.0) or 0.0)
        px = broker_px if broker_px > 0 else _last_price(cur, sym, fallback=0.0)

        if delta > 0:
            _insert_trade(
                cur,
                symbol=sym,
                side="buy",
                qty=delta,
                price=px,
                reason=f"AUTO_HEAL_SYNC_BUY_{reason}",
            )
            changes.append(f"{sym}:+{delta}")
        else:
            _insert_trade(
                cur,
                symbol=sym,
                side="sell",
                qty=abs(delta),
                price=px,
                reason=f"AUTO_HEAL_SYNC_SELL_{reason}",
            )
            changes.append(f"{sym}:{delta}")

    conn.commit()
    conn.close()

    logger.warning(
        f"AUTO-HEAL APPLIED — DB positions synced to broker ({reason}) "
        f"{'changes=' + ','.join(changes) if changes else 'changes=none'}"
    )
