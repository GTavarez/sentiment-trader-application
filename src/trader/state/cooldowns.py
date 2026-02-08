import json
from pathlib import Path
from datetime import datetime

COOLDOWN_FILE = Path("data/cooldowns.json")

def load_cooldowns():
    if not COOLDOWN_FILE.exists():
        return {}
    data = json.loads(COOLDOWN_FILE.read_text())
    return {k: datetime.fromisoformat(v) for k, v in data.items()}

def save_cooldowns(cooldowns: dict):
    COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v.isoformat() for k, v in cooldowns.items()}
    COOLDOWN_FILE.write_text(json.dumps(serializable, indent=2))
