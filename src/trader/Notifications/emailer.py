import smtplib
from email.message import EmailMessage
from loguru import logger
import time

_email_disabled = False


def send_email(subject: str, body: str, settings):
    """
    Send an email alert using SMTP credentials from settings.
    """
    global _email_disabled

    if not getattr(settings, "send_emails", True):
        return

    if _email_disabled:
        return

    subject = subject.replace("blockedd", "blocked").replace("bloccked", "blocked")

    msg = EmailMessage()
    msg["From"] = settings.email_from
    msg["To"] = settings.email_to
    msg["Subject"] = subject
    msg.set_content(body)

    max_retries = max(1, int(getattr(settings, "email_max_retries", 2)))
    base_s = max(0.0, float(getattr(settings, "email_retry_base_s", 0.5)))
    max_s = max(0.0, float(getattr(settings, "email_retry_max_s", 5.0)))

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with smtplib.SMTP(settings.smtp_server, settings.smtp_port) as server:
                server.starttls()
                server.login(settings.email_from, settings.email_password)
                server.send_message(msg)

            logger.info(f"Email sent: {subject}")
            return
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                break
            sleep_s = min(base_s * (2 ** (attempt - 1)), max_s)
            logger.warning(
                f"Email send failed (attempt {attempt}/{max_retries}): {e}"
            )
            if sleep_s > 0:
                time.sleep(sleep_s)

    logger.error(f"Failed to send email: {last_err}")
    _email_disabled = True
    logger.warning("Email disabled for this run after repeated failures.")
