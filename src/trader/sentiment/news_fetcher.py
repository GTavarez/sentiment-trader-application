import requests
import re
from loguru import logger


def is_relevant_headline(headline: str, symbol: str) -> bool:
    text = headline.lower()

    # ---- must mention company or ticker ----
    keywords = [
        symbol.lower(),
        "apple",
        "iphone",
        "ipad",
        "mac",
        "ios",
        "siri",
        "tim cook",
    ]

    if not any(k in text for k in keywords):
        return False

    # ---- exclude obvious macro / unrelated ----
    blacklist = [
        "dow",
        "nasdaq",
        "s&p",
        "trump",
        "fed",
        "interest rate",
        "401",
        "crypto",
        "bitcoin",
        "oil",
        "gold",
    ]

    if any(b in text for b in blacklist):
        return False

    # ---- basic english check ----
    if not re.match(r"^[a-zA-Z0-9\s\.,'’\"!?\-:;$%()]+$", headline):
        return False

    return True


class NewsFetcher:
    """
    Fetches recent news headlines for a given symbol using NewsAPI.
    Applies relevance filtering before returning headlines.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch(self, symbol: str, limit: int = 5) -> list[str]:
        params = {
            "q": symbol,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": limit,
            "apiKey": self.api_key,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])
            raw_headlines = [
                article["title"]
                for article in articles
                if article.get("title")
            ]

            filtered = [
                h for h in raw_headlines
                if is_relevant_headline(h, symbol)
            ]

            logger.info(
                f"News filter | raw={len(raw_headlines)} filtered={len(filtered)}"
            )

            return filtered

        except requests.RequestException as e:
            logger.error(f"News API request failed for {symbol}: {e}")
            return []

        except Exception as e:
            logger.error(f"Unexpected error fetching news for {symbol}: {e}")
            return []
