# src/trader/state/halt_state.py
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

HALT_FILE = Path("data/trading_halt.json")
UNBLOCK_FILE = Path("data/unblock_ack.json")
UNBLOCK_ACK_FILE = Path("data/unblock_ack.json")


def _ensure_data_dir() -> None:
    HALT_FILE.parent.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def compute_fingerprint(recon_summary: Dict[str, Any]) -> str:
    """
    Stable fingerprint of the reconciliation result.
    If this changes, the operator must re-ack/unblock.
    """
    payload = json.dumps(recon_summary, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class HaltState:
    is_halted: bool
    reason: str
    details: Dict[str, Any]
    created_at: str
    fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_halted": self.is_halted,
            "reason": self.reason,
            "details": self.details,
            "created_at": self.created_at,
            "fingerprint": self.fingerprint,
        }


def load_halt_state() -> Optional[HaltState]:
    if not HALT_FILE.exists():
        return None
    try:
        raw = json.loads(HALT_FILE.read_text())
        return HaltState(
            is_halted=bool(raw.get("is_halted", False)),
            reason=str(raw.get("reason", "")),
            details=dict(raw.get("details", {})),
            created_at=str(raw.get("created_at", "")),
            fingerprint=str(raw.get("fingerprint", "")),
        )
    except Exception:
        return None


def write_halt(reason: str, details: Dict[str, Any]) -> HaltState:
    _ensure_data_dir()
    fingerprint = compute_fingerprint(details)
    state = HaltState(
        is_halted=True,
        reason=reason,
        details=details,
        created_at=_utc_now_iso(),
        fingerprint=fingerprint,
    )
    HALT_FILE.write_text(json.dumps(state.to_dict(), indent=2))
    return state


def clear_halt() -> None:
    if HALT_FILE.exists():
        HALT_FILE.unlink(missing_ok=True)
    if UNBLOCK_FILE.exists():
        UNBLOCK_FILE.unlink(missing_ok=True)


def load_unblock_ack() -> Dict[str, Any]:
    if not UNBLOCK_FILE.exists():
        return {}
    try:
        return json.loads(UNBLOCK_FILE.read_text())
    except Exception:
        return {}


def write_unblock_ack(fingerprint: str, note: str = "") -> None:
    _ensure_data_dir()
    payload = {
        "fingerprint": fingerprint,
        "note": note,
        "ack_at": _utc_now_iso(),
    }
    UNBLOCK_FILE.write_text(json.dumps(payload, indent=2))


def is_unblocked_for(fingerprint: str) -> bool:
    """Return True if the unblock_ack.json exists and fingerprint matches."""
    if not UNBLOCK_ACK_FILE.exists():
        return False
    try:
        data = json.loads(UNBLOCK_ACK_FILE.read_text())
        return data.get("fingerprint") == fingerprint
    except Exception:
        return False
