from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    env: str = "dev"

    broker: str = "alpaca"
    trading_mode: str = "paper"
    alpaca_api_key: str = Field(..., env="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(..., env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(..., env="ALPACA_BASE_URL")
    news_api_key: str = Field(..., env="NEWS_API_KEY")

    max_trades_per_day: int = 5
    max_position_usd: float = 50.0
    daily_loss_limit_usd: float = 20.0

    buy_threshold: float = 0.6
    sell_threshold: float = -0.6
      # ---- EMAIL ----
    email_from: str
    email_to: str
    email_password: str
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
