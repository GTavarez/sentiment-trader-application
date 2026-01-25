class SentimentStrategy:
    def __init__(self, buy_threshold: float, sell_threshold: float):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def decide(self, sentiment_score: float) -> str:
        if sentiment_score >= self.buy_threshold:
            return "buy"
        if sentiment_score <= self.sell_threshold:
            return "sell"
        return "hold"
