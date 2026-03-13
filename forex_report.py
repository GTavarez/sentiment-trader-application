import sqlite3
from pathlib import Path


DB_PATH = Path("data/forex_trader.db")


def main() -> None:
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    total = cur.execute("SELECT COUNT(*) FROM fx_trades WHERE pnl IS NOT NULL").fetchone()[0]
    wins = cur.execute("SELECT COUNT(*) FROM fx_trades WHERE pnl > 0").fetchone()[0]
    losses = cur.execute("SELECT COUNT(*) FROM fx_trades WHERE pnl < 0").fetchone()[0]
    pnl = cur.execute("SELECT COALESCE(SUM(pnl),0) FROM fx_trades WHERE pnl IS NOT NULL").fetchone()[0]
    avg_win = cur.execute("SELECT COALESCE(AVG(pnl),0) FROM fx_trades WHERE pnl > 0").fetchone()[0]
    avg_loss = cur.execute("SELECT COALESCE(AVG(pnl),0) FROM fx_trades WHERE pnl < 0").fetchone()[0]

    win_rate = (wins / total * 100.0) if total else 0.0
    ratio = (avg_win / abs(avg_loss)) if avg_loss else 0.0
    expectancy = (pnl / total) if total else 0.0

    print("=== Forex Summary ===")
    print(f"Closed trades: {total}")
    print(f"Wins: {wins} | Losses: {losses} | Win rate: {win_rate:.1f}%")
    print(f"Avg win: {avg_win:.2f} | Avg loss: {avg_loss:.2f} | Win/Loss ratio: {ratio:.2f}")
    print(f"Expectancy: {expectancy:.2f} | Total realized PnL: {pnl:.2f}")
    print("")

    print("=== By Pair ===")
    rows = cur.execute(
        """
        SELECT
            pair,
            COUNT(*) AS n,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losses,
            COALESCE(AVG(pnl), 0) AS expectancy,
            COALESCE(SUM(pnl), 0) AS total_pnl
        FROM fx_trades
        WHERE pnl IS NOT NULL
        GROUP BY pair
        ORDER BY total_pnl ASC
        """
    ).fetchall()
    for pair, n, w, l, ex, tpnl in rows:
        wr = (w / n * 100.0) if n else 0.0
        print(f"{pair}: n={n}, wr={wr:.1f}%, exp={ex:.2f}, pnl={tpnl:.2f}")

    conn.close()


if __name__ == "__main__":
    main()
