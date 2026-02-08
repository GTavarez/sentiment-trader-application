# src/trader/state/reconciliation.py
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
from rich import print

from src.trader.state.db_positions import load_latest_positions


def reconcile_positions(broker) -> Dict[str, Any]:
    """
    Compare DB positions vs. live broker positions.

    Returns a dict:
    {
        ok: bool,
        summary: {qty_mismatch, ghost_db, ghost_broker},
        rows: [{symbol, db_qty, broker_qty, qty_diff, status}]
    }
    """

    # ------------------
    # DB positions
    # ------------------
    db_df = load_latest_positions()
    if db_df is None or db_df.empty:
        db_df = pd.DataFrame(columns=["symbol", "qty"])
    else:
        db_df.columns = [c.lower() for c in db_df.columns]  # normalize case
        if "qty" not in db_df.columns:
            # attempt to map alternate qty field names
            possible_qty_cols = ["quantity", "shares", "position", "amount"]
            for alt in possible_qty_cols:
                if alt in db_df.columns:
                    db_df = db_df.rename(columns={alt: "qty"})
                    break
        if "qty" not in db_df.columns:
            raise KeyError(f"DB positions missing qty column: {db_df.columns.tolist()}")
        db_df = db_df[["symbol", "qty"]]

    print("DB POSITIONS LOADED:", db_df.columns.tolist())

    # ------------------
    # Broker positions
    # ------------------
    broker_positions = broker.get_positions()
    print("Sample broker_positions (first 1):", broker_positions[:1])

    broker_rows: List[Dict[str, Any]] = []

    if broker_positions:
        for p in broker_positions:
            if hasattr(p, "symbol") and hasattr(p, "qty"):
                sym = p.symbol
                qty = int(p.qty)
            elif isinstance(p, dict):
                sym = p.get("symbol") or p.get("ticker") or "UNKNOWN"
                qty = int(p.get("qty") or p.get("quantity") or 0)
            else:
                raise TypeError(f"Unexpected broker position type: {type(p)}")
            broker_rows.append({"symbol": sym, "qty": qty})

    # always ensure broker_df has proper columns
    broker_df = pd.DataFrame(broker_rows, columns=["symbol", "qty"])

    # ------------------
    # Merge + compare
    # ------------------
    merged = pd.merge(
        db_df,
        broker_df,
        on="symbol",
        how="outer",
        suffixes=("_db", "_broker"),
    )

    merged["qty_db"] = pd.to_numeric(merged["qty_db"], errors="coerce").fillna(0).astype(int)
    merged["qty_broker"] = pd.to_numeric(merged["qty_broker"], errors="coerce").fillna(0).astype(int)
    merged["qty_diff"] = merged["qty_db"] - merged["qty_broker"]

    rows: List[Dict[str, Any]] = []
    qty_mismatch = ghost_db = ghost_broker = 0

    for _, r in merged.iterrows():
        status = "MATCH"
        if r["qty_db"] == 0 and r["qty_broker"] != 0:
            status = "GHOST_BROKER_POSITION"
            ghost_broker += 1
        elif r["qty_db"] != 0 and r["qty_broker"] == 0:
            status = "GHOST_DB_POSITION"
            ghost_db += 1
        elif r["qty_db"] != r["qty_broker"]:
            status = "QTY_MISMATCH"
            qty_mismatch += 1

        rows.append({
            "symbol": r["symbol"],
            "db_qty": int(r["qty_db"]),
            "broker_qty": int(r["qty_broker"]),
            "qty_diff": int(r["qty_diff"]),
            "status": status,
        })

    ok = qty_mismatch == 0 and ghost_db == 0 and ghost_broker == 0

    if ok:
        print("[green]✅ Reconciliation OK — DB and broker positions match.[/green]")
    else:
        print("[red]🚨 Reconciliation mismatch detected![/red]")
        print("Reconciliation summary:", {
            "qty_mismatch": qty_mismatch,
            "ghost_db": ghost_db,
            "ghost_broker": ghost_broker,
        })
        print("Reconciliation rows:")
        for row in rows:
            print(row)

    return {
        "ok": ok,
        "summary": {
            "qty_mismatch": qty_mismatch,
            "ghost_db": ghost_db,
            "ghost_broker": ghost_broker,
        },
        "rows": rows,
    }
