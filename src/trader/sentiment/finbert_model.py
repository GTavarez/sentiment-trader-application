from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from loguru import logger


class FinBertSentimentModel:
    def __init__(self):
        logger.info("Loading FinBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ProsusAI/finbert"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        )
        self.model.eval()

    def score_texts(self, texts: list[str]) -> float:
        if not texts:
            return 0.0

        scores = []

        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

            # FinBERT labels: [negative, neutral, positive]
            negative, neutral, positive = probs[0].tolist()
            score = positive - negative
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        return avg_score
