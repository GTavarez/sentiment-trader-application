import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw.strip())


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


@dataclass
class ForexSettings:
    mode: str = os.getenv("FOREX_MODE", "paper").strip().lower()
    pairs: list[str] = None  # type: ignore[assignment]
    signal_threshold_pct: float = _float_env("FOREX_SIGNAL_THRESHOLD_PCT", 0.002)
    notional_usd: float = _float_env("FOREX_NOTIONAL_USD", 1000.0)
    max_hold_days: int = _int_env("FOREX_MAX_HOLD_DAYS", 3)
    read_only: bool = _bool_env("FOREX_READ_ONLY", False)
    api_base: str = os.getenv("FOREX_API_BASE", "https://api.frankfurter.app").strip()
    db_path: Path = Path(os.getenv("FOREX_DB_PATH", "data/forex_trader.db"))

    def __post_init__(self) -> None:
        raw_pairs = os.getenv("FOREX_PAIRS", "USD/CAD,EUR/USD,GBP/USD,USD/JPY")
        self.pairs = [p.strip().upper() for p in raw_pairs.split(",") if p.strip()]


def ensure_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fx_positions (
            pair TEXT PRIMARY KEY,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            entry_price REAL NOT NULL,
            entry_ts TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fx_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            pair TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            reason TEXT,
            pnl REAL
        )
        """
    )
    conn.commit()
    conn.close()


def _http_json(url: str) -> dict:
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError:
        # Retry once with a simpler user-agent for stricter edge gateways.
        req = Request(url, headers={"User-Agent": "python-forex-bot/1.0", "Accept": "application/json"})
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))


def fetch_rate(api_base: str, pair: str, d: Optional[date] = None) -> tuple[str, float]:
    base, quote = pair.split("/")
    day_path = "latest" if d is None else d.isoformat()
    url = f"{api_base}/{day_path}?from={base}&to={quote}"
    payload = _http_json(url)
    rate = float(payload["rates"][quote])
    asof = str(payload["date"])
    return asof, rate


def previous_business_day(today: date) -> date:
    d = today - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def load_position(conn: sqlite3.Connection, pair: str) -> Optional[tuple[str, float, float, str]]:
    row = conn.execute(
        "SELECT side, qty, entry_price, entry_ts FROM fx_positions WHERE pair = ?",
        (pair,),
    ).fetchone()
    if not row:
        return None
    return str(row[0]), float(row[1]), float(row[2]), str(row[3])


def save_position(
    conn: sqlite3.Connection, pair: str, side: str, qty: float, entry_price: float, entry_ts: str
) -> None:
    conn.execute(
        """
        INSERT INTO fx_positions(pair, side, qty, entry_price, entry_ts)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(pair) DO UPDATE SET
            side=excluded.side,
            qty=excluded.qty,
            entry_price=excluded.entry_price,
            entry_ts=excluded.entry_ts
        """,
        (pair, side, qty, entry_price, entry_ts),
    )


def delete_position(conn: sqlite3.Connection, pair: str) -> None:
    conn.execute("DELETE FROM fx_positions WHERE pair = ?", (pair,))


def log_trade(
    conn: sqlite3.Connection,
    pair: str,
    side: str,
    qty: float,
    price: float,
    reason: str,
    pnl: Optional[float],
) -> None:
    conn.execute(
        """
        INSERT INTO fx_trades(timestamp, pair, side, qty, price, reason, pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (datetime.now(timezone.utc).isoformat(), pair, side, qty, price, reason, pnl),
    )


def run_once(settings: ForexSettings) -> None:
    if settings.mode != "paper":
        raise RuntimeError("FOREX_MODE must be 'paper' for this bot.")

    ensure_db(settings.db_path)
    conn = sqlite3.connect(settings.db_path)
    conn.isolation_level = None
    conn.execute("BEGIN")
    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()
    prev_day = previous_business_day(today)

    print(f"Forex bot mode: {settings.mode.upper()}")
    print(f"Pairs: {', '.join(settings.pairs)}")
    print(f"Threshold: {settings.signal_threshold_pct:.4f}")
    print(f"Notional USD: {settings.notional_usd:.2f}")
    print(f"Read-only: {settings.read_only}")
    print("")

    try:
        for pair in settings.pairs:
            try:
                asof_now, px_now = fetch_rate(settings.api_base, pair)
                asof_prev, px_prev = fetch_rate(settings.api_base, pair, prev_day)
            except (URLError, KeyError, ValueError) as e:
                print(f"{pair}: data fetch failed ({e})")
                continue

            change_pct = (px_now - px_prev) / px_prev if px_prev > 0 else 0.0
            signal = "hold"
            if change_pct >= settings.signal_threshold_pct:
                signal = "long"
            elif change_pct <= -settings.signal_threshold_pct:
                signal = "short"

            pos = load_position(conn, pair)
            print(
                f"{pair}: rate={px_now:.6f} ({asof_now}) prev={px_prev:.6f} ({asof_prev}) "
                f"chg={change_pct*100:.3f}% signal={signal}"
            )

            if pos is None:
                if signal in {"long", "short"}:
                    qty = settings.notional_usd / px_now
                    if settings.read_only:
                        print(f"  READ-ONLY: would open {signal} qty={qty:.4f}")
                    else:
                        save_position(conn, pair, signal, qty, px_now, now_utc.isoformat())
                        entry_side = "long_entry" if signal == "long" else "short_entry"
                        log_trade(conn, pair, entry_side, qty, px_now, "signal_entry", None)
                        print(f"  OPENED {signal} qty={qty:.4f} @ {px_now:.6f}")
                continue

            side, qty, entry_price, entry_ts = pos
            held_days = 0
            try:
                held_days = max(0, (today - datetime.fromisoformat(entry_ts).date()).days)
            except Exception:
                held_days = 0

            opposite_signal = (side == "long" and signal == "short") or (
                side == "short" and signal == "long"
            )
            timed_exit = held_days >= settings.max_hold_days
            should_exit = opposite_signal or timed_exit

            if not should_exit:
                continue

            reason = "opposite_signal" if opposite_signal else "max_hold_days"
            pnl = (px_now - entry_price) * qty if side == "long" else (entry_price - px_now) * qty
            if settings.read_only:
                print(f"  READ-ONLY: would close {side} qty={qty:.4f} pnl={pnl:.2f} ({reason})")
            else:
                delete_position(conn, pair)
                exit_side = "long_exit" if side == "long" else "short_exit"
                log_trade(conn, pair, exit_side, qty, px_now, reason, pnl)
                print(f"  CLOSED {side} qty={qty:.4f} @ {px_now:.6f} pnl={pnl:.2f} ({reason})")

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run_once(ForexSettings())
