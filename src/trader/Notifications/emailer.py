import smtplib
from email.message import EmailMessage
from loguru import logger


def send_email(subject: str, body: str, settings):
    """
    Send an email alert using SMTP credentials from settings.
    """

    msg = EmailMessage()
    msg["From"] = settings.email_from
    msg["To"] = settings.email_to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(settings.smtp_server, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.email_from, settings.email_password)
            server.send_message(msg)

        logger.info(f"Email sent: {subject}")

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
