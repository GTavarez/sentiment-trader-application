from datetime import datetime
from typing import Dict, Any


def can_auto_heal(*, trading_mode: str, summary: Dict[str, int]) -> bool:
    """
    Allow auto-heal ONLY if:
    - PAPER mode
    - No ghost broker positions
    - No quantity mismatches
    """

    if trading_mode.lower() != "paper":
        return False

    if summary.get("qty_mismatch", 0) != 0:
        return False

    return True


def auto_heal_action(summary: Dict[str, int]) -> Dict[str, Any]:
    """
    Decide what healing action to take.

    IMPORTANT:
    - This function ONLY decides.
    - Execution happens elsewhere.
    """

    if summary.get("ghost_db", 0) > 0 and summary.get("ghost_broker", 0) > 0:
        return {
            "action": "SYNC_DB_WITH_BROKER",
            "reason": "DB and broker both have ghost positions",
            "timestamp": datetime.utcnow().isoformat(),
        }

    if summary.get("ghost_db", 0) > 0:
        return {
            "action": "CLEAR_DB_POSITIONS",
            "reason": "DB has positions broker does not",
            "timestamp": datetime.utcnow().isoformat(),
        }

    if summary.get("ghost_broker", 0) > 0:
        return {
            "action": "REBUILD_DB_FROM_BROKER",
            "reason": "Broker has positions DB does not",
            "timestamp": datetime.utcnow().isoformat(),
        }

    return {
        "action": "NONE",
        "reason": "No safe auto-heal path",
        "timestamp": datetime.utcnow().isoformat(),
    }
