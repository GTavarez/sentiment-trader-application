import argparse
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


LOG_PATH = Path("forex_output.log")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _tail_log_lines(limit: int = 40) -> str:
    if not LOG_PATH.exists():
        return "forex_output.log not found."
    try:
        lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"failed reading forex_output.log: {exc}"
    tail = lines[-limit:] if lines else []
    return "\n".join(tail) if tail else "(log empty)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc", type=int, default=1, help="Failed process exit code")
    args = parser.parse_args()

    if not _env_bool("FOREX_ALERTS_ENABLED", True):
        print("forex alert skipped: FOREX_ALERTS_ENABLED=false")
        return 0

    if not _env_bool("SEND_EMAILS", True):
        print("forex alert skipped: SEND_EMAILS=false")
        return 0

    if _env_bool("FOREX_ALERT_DRY_RUN", False):
        print("forex alert dry-run enabled; email not sent")
        print(f"[ALERT] Forex bot failed (rc={args.rc})")
        return 0

    email_from = os.getenv("EMAIL_FROM", "").strip()
    email_to = os.getenv("EMAIL_TO", "").strip()
    email_password = os.getenv("EMAIL_PASSWORD", "").strip()
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    missing = [
        k
        for k, v in {
            "EMAIL_FROM": email_from,
            "EMAIL_TO": email_to,
            "EMAIL_PASSWORD": email_password,
        }.items()
        if not v
    ]
    if missing:
        print(f"forex alert skipped: missing env vars: {', '.join(missing)}")
        return 0

    subject = f"[ALERT] Forex bot failed (rc={args.rc})"
    body = (
        "Forex scheduled run failed.\n\n"
        f"Exit code: {args.rc}\n"
        f"Host: {os.getenv('COMPUTERNAME', 'unknown')}\n\n"
        "Last log lines:\n"
        "----------------\n"
        f"{_tail_log_lines(40)}\n"
    )

    msg = EmailMessage()
    msg["From"] = email_from
    msg["To"] = email_to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_from, email_password)
            server.send_message(msg)
        print("forex alert email sent")
    except Exception as exc:
        # Do not turn alert failure into a different bot status.
        print(f"forex alert send failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
