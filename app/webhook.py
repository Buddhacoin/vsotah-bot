"""Webhook helper notes for VSotahBot.

The real webhook endpoint is registered in app.bot_app so all existing aiogram
handlers keep working without moving code around.

Railway variables for webhook mode:
    WEBHOOK_MODE=true
    WEBHOOK_URL=https://<your-service>.up.railway.app
    WEBHOOK_SECRET=<random-long-secret>  # optional but recommended
    WEBHOOK_DROP_PENDING_UPDATES=false

To rollback instantly:
    WEBHOOK_MODE=false
"""

from __future__ import annotations

import secrets


def generate_webhook_secret() -> str:
    """Generate a strong Telegram webhook secret token."""
    return secrets.token_urlsafe(32)
