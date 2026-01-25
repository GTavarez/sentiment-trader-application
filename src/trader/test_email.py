from src.trader.config import settings
from src.trader.Notifications.emailer import send_email

send_email(
    subject="✅ Sentiment Trader Email Test",
    body="If you see this, email alerts are working.",
    settings=settings,
)
