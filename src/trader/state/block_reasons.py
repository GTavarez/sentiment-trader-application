from pathlib import Path
from datetime import datetime
import json

BLOCK_FILE = Path("data/block_reasons.json")

def block_symbol(symbol: str, reason: str):
    data = {}
    if BLOCK_FILE.exists():
        data = json.loads(BLOCK_FILE.read_text())

    data[symbol] = {
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat(),
    }

    BLOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    BLOCK_FILE.write_text(json.dumps(data, indent=2))
