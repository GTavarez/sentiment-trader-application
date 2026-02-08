import json
from pathlib import Path
from datetime import datetime

BLOCK_FILE = Path("data/block_reasons.json")

def load_block_reasons():
    if not BLOCK_FILE.exists():
        return {}
    return json.loads(BLOCK_FILE.read_text())

def save_block_reason(symbol: str, reason: str):
    BLOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = load_block_reasons()
    data[symbol] = {
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat(),
    }
    BLOCK_FILE.write_text(json.dumps(data, indent=2))
