"""Webhook helpers for VSotahBot.

The production webhook route is registered from app.bot_app without replacing
existing handlers. This file is intentionally small: it documents the ENV
variables and keeps webhook-related architecture explicit.

ENV:
- WEBHOOK_MODE=true
- WEBHOOK_URL=https://your-railway-domain.up.railway.app
- WEBHOOK_PATH=/telegram-webhook  # optional
- WEBHOOK_SECRET=any-random-secret # optional, recommended later
"""

DEFAULT_WEBHOOK_PATH = "/telegram-webhook"
