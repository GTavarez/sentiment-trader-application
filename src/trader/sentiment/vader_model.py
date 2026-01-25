from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderSentimentModel:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def score_texts(self, texts: list[str]) -> float:
        """
        Returns aggregated compound sentiment score in range [-1, 1]
        """
        if not texts:
            return 0.0

        scores = []
        for text in texts:
            result = self.analyzer.polarity_scores(text)
            scores.append(result["compound"])

        return sum(scores) / len(scores)
