import json
from datetime import datetime
from pathlib import Path
from typing import List

SYMBOLS_FILE = Path("data/symbols.json")


def load_symbols(defaults: List[str]) -> List[str]:
    if not SYMBOLS_FILE.exists():
        return [s.upper() for s in defaults]
    try:
        raw = json.loads(SYMBOLS_FILE.read_text())
        symbols = raw.get("symbols", [])
        return [str(s).upper() for s in symbols if str(s).strip()]
    except Exception:
        return [s.upper() for s in defaults]


def save_symbols(symbols: List[str]) -> None:
    SYMBOLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbols": [s.upper() for s in symbols],
        "updated_at": datetime.utcnow().isoformat(),
    }
    SYMBOLS_FILE.write_text(json.dumps(payload, indent=2))
