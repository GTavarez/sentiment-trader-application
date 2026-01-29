from datetime import datetime
from typing import List


class SentimentStrategy:
    def __init__(
        self,
        buy_threshold: float,
        sell_threshold: float,
        max_confidence: float = 1.0,
        decay_half_life_hours: float = 24.0,
        min_confidence: float = 0.2,
    ):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.max_confidence = max_confidence
        self.decay_half_life_hours = decay_half_life_hours
        self.min_confidence = min_confidence



    def decide(
        self,
        sentiment_score: float,
        timestamps: List[datetime] | None = None,
    ) -> dict:
        score = self._apply_decay(sentiment_score, timestamps)
        confidence = self._confidence_from_score(score)

        if score >= self.buy_threshold:
            decision = "buy"
        elif score <= self.sell_threshold:
            decision = "sell"
        else:
            return self._hold()

        return {
            "decision": decision,
            "confidence": round(confidence, 3),
            "position_scale": round(confidence, 3),
        }

    def _apply_decay(self, score, timestamps):
        if not timestamps:
            return score

        now = datetime.utcnow()
        ages = [(now - ts).total_seconds() / 3600 for ts in timestamps]
        weights = [0.5 ** (age / self.decay_half_life_hours) for age in ages]

        if sum(weights) == 0:
            return 0.0

        return score * (sum(weights) / len(weights))

    def _confidence_from_score(self, score):
        abs_score = abs(score)
        return max(self.min_confidence, min(self.max_confidence, abs_score))

    def _hold(self):
        return {
            "decision": "hold",
            "confidence": 0.0,
            "position_scale": 0.0,
        }
