import json
from pathlib import Path

FILE = Path("sentiment_streaks.json")

def load_streaks():
    if FILE.exists():
        return json.loads(FILE.read_text())
    return {}

def save_streaks(streaks):
    FILE.write_text(json.dumps(streaks))
