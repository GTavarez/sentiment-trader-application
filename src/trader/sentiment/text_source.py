class TextSource:
    """
    MVP placeholder for sentiment input.
    In Step 1.5 we'll replace this with real news.
    """

    def fetch(self, symbol: str) -> list[str]:
        if symbol == "AAPL":
            return [
                "Apple stock rises as earnings beat expectations",
                "Strong iPhone sales drive Apple growth",
                "Analysts raise Apple price targets",
            ]

        return []
