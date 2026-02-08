from typing import Dict, List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---- Broker ----
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_api_key_paper: str = ""
    alpaca_secret_key_paper: str = ""
    alpaca_api_key_live: str = ""
    alpaca_secret_key_live: str = ""

    # ---- Trading ----
    trading_mode: str = "paper"
    live_trading_confirm: bool = False
    live_trading_confirm_code: str = ""
    read_only: bool = False

    # Multi-symbol support
    symbol_list: List[str] = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    max_symbols_per_run: int = 0

    max_trades_per_day: int = 5
    max_position_usd: float = 500
    max_position_usd_by_symbol: Dict[str, float] = {}
    max_symbol_exposure_usd: float = 500
    max_symbol_exposure_by_symbol: Dict[str, float] = {}
    daily_loss_limit_usd: float = 300
    buy_threshold: float = 0.02
    sell_threshold: float = -0.02
    cooldown_minutes: int = 30
    per_symbol_cooldown_minutes: Dict[str, int] = {}
    allow_scale_in: bool = False
    send_daily_summary_email: bool = False
    send_scale_in_block_email: bool = False
    send_daily_health_email: bool = False
    send_startup_email_test: bool = False
    send_emails: bool = True
    email_max_retries: int = 2
    email_retry_base_s: float = 0.5
    email_retry_max_s: float = 5.0
    min_signal_cycles: int = 2
    trend_filter_enabled: bool = True
    trend_sma_days: int = 20
    volatility_filter_enabled: bool = True
    volatility_lookback_days: int = 20
    max_daily_volatility_pct: float = 0.03
    apply_filters_to_sell: bool = True
    take_profit_pct: float = 0.03
    stop_loss_pct: float = 0.02
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = 0.02

    # ---- Reliability ----
    last_price_max_retries: int = 3
    last_price_retry_base_s: float = 0.5
    last_price_retry_max_s: float = 5.0

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
