import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path("data/trader.db")
def load_latest_positions() -> pd.DataFrame:
    """
    Reconstruct current positions from the trades table.
    Buys add quantity, sells subtract quantity.
    Returns per-symbol qty, avg_cost, and placeholder market_value.
    """
    # ------------------------------------------------------------
    # 1.  Load trades table
    # ------------------------------------------------------------
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["symbol", "qty", "avg_cost", "market_value"])
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT timestamp, symbol, side, qty, price
        FROM trades
        ORDER BY timestamp ASC
        """,
        conn,
    )
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=["symbol", "qty", "avg_cost", "market_value"])
    # ------------------------------------------------------------
    # 2.  Normalize column types
    # ------------------------------------------------------------
    df["side"] = df["side"].astype(str).str.lower()
    # Convert qty/price to proper numerics (SQLite stores as text)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["qty", "price"])
    # ------------------------------------------------------------
    # 3.  Signed quantities
    # ------------------------------------------------------------
    df["signed_qty"] = df.apply(
    lambda r: int(r["qty"]) if str(r["side"]).lower() == "buy" else -int(r["qty"]),
    axis=1,
)

    # ------------------------------------------------------------
    # 4.  Net quantity per symbol
    # ------------------------------------------------------------
    qty_by_symbol = (
        df.groupby("symbol", as_index=False)["signed_qty"]
        .sum()
        .rename(columns={"signed_qty": "qty"})
    )
    # ------------------------------------------------------------
    # 5.  Average cost per symbol (buys only)
    # ------------------------------------------------------------
    buys = df[df["side"] == "buy"].copy()
    if buys.empty:
        qty_by_symbol["avg_cost"] = 0.0
    else:
        buys["notional"] = buys["qty"] * buys["price"]
        # vectorized avg cost, using include_groups=False to silence pandas warning
        avg_cost = (
            buys.groupby("symbol", as_index=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "avg_cost": g["notional"].sum() / g["qty"].sum()
                        if g["qty"].sum()
                        else 0.0
                    }
                ),
                include_groups=False,
            )
        )
        qty_by_symbol = qty_by_symbol.merge(avg_cost, on="symbol", how="left")
    qty_by_symbol["avg_cost"] = qty_by_symbol["avg_cost"].fillna(0.0)
    # ------------------------------------------------------------
    # 6.  Keep open positions only (qty > 0)
    # ------------------------------------------------------------
    qty_by_symbol = qty_by_symbol[qty_by_symbol["qty"] > 0].reset_index(drop=True)
    # Placeholder for market_value (will be filled in later by broker prices)
    qty_by_symbol["market_value"] = 0.0
    # ------------------------------------------------------------
    # 7.  Sort for readability
    # ------------------------------------------------------------
    qty_by_symbol = qty_by_symbol.sort_values("qty", ascending=False).reset_index(
        drop=True
    )
    return qty_by_symbol