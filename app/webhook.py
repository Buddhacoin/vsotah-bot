"""
Webhook helper module for VSotahBot.

The active webhook route is intentionally registered inside app.bot_app.start_web_server()
so the existing Telegram handlers, voice pipeline, loading animation, referrals,
payments, and /start behavior stay unchanged.
"""

import os


def webhook_enabled() -> bool:
    return os.getenv("WEBHOOK_MODE", "false").lower() in {"1", "true", "yes", "on"}


def webhook_url() -> str:
    return os.getenv("WEBHOOK_URL", "").rstrip("/")
