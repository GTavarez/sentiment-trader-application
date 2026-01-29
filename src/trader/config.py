from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- Broker ----
    alpaca_api_key: str
    alpaca_secret_key: str

    # ---- Trading ----
    trading_mode: str = "paper"

    # Multi-symbol support
    symbol_list: List[str] = ["AAPL", "MSFT", "NVDA"]

    max_trades_per_day: int = 5
    max_position_usd: float = 500
    daily_loss_limit_usd: float = 300
    buy_threshold: float = 0.02
    sell_threshold: float = -0.02
    cooldown_minutes: int = 30

    # ---- News ----
    news_api_key: str

    # ---- Email ----
    email_from: str
    email_to: str
    email_password: str
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587

    # ✅ Pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_delimiter=",",
        case_sensitive=False,
        extra="ignore",
    )
    

settings = Settings()
