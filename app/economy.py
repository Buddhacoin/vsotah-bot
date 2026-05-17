"""VSotah AI economy configuration.

This module keeps tariffs, Stars prices, VS token packs and fair-use rules
outside app/bot_app.py so the Telegram layer does not contain business logic.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

FREE_DAILY_LIMIT = 14
FREE_WEEKLY_LIMIT = 98
FREE_DAILY_IMAGE_LIMIT = 3

PLAN_DAILY_LIMITS: dict[str, int | None] = {
    "FREE": 14,
    "PLUS": 100,
    "PRO": 200,
}

# Kept for old checks/admin screens that still read weekly counters.
PLAN_WEEKLY_LIMITS: dict[str, int | None] = {
    "FREE": 98,
    "PLUS": 700,
    "PRO": 1400,
}

TARIFFS: dict[str, dict[str, Any]] = {
    "PLUS": {
        "title": "⭐ PLUS",
        "short_title": "PLUS",
        "daily_limit": 100,
        "bonus_tokens": 100,
        "description": (
            "до 100 запросов в день\n"
            "• последние AI-модели\n"
            "• работа с документами\n"
            "• голосовые ответы\n"
            "• разумное использование\n"
            "• без тяжёлых моделей глубокого анализа"
        ),
        "prices": {1: 600, 3: 1200, 6: 2000, 12: 3000},
        "rub_prices": {1: 850, 3: 1750, 6: 2850, 12: 4350},
    },
    "PRO": {
        "title": "💎 PRO",
        "short_title": "PRO",
        "daily_limit": 200,
        "bonus_tokens": 300,
        "description": (
            "до 200 запросов в день\n"
            "• всё из PLUS\n"
            "• GPT-5 чат\n"
            "• Claude\n"
            "• память\n"
            "• файлы\n"
            "• приоритет\n"
            "• длинный контекст и кодинг"
        ),
        "prices": {1: 900, 3: 1800, 6: 3000, 12: 4500},
        "rub_prices": {1: 1300, 3: 2600, 6: 4350, 12: 6500},
    },
}

VS_TOKEN_PACKS: dict[int, dict[str, int]] = {
    100: {"stars": 140, "rub": 199},
    200: {"stars": 255, "rub": 365},
    500: {"stars": 590, "rub": 849},
    800: {"stars": 890, "rub": 1280},
    1000: {"stars": 1050, "rub": 1499},
}

# Internal default costs. They can be tuned later when exact providers are final.
VS_TOKEN_COSTS: dict[str, int] = {
    # Image generation/editing
    "nanobanana_generate": 8,
    "nanobanana_edit": 10,
    "gptimage_generate": 10,
    "gptimage_edit": 12,
    "image_default": 10,
    "image_edit_default": 12,

    # Future image/video providers
    "flux_schnell_min": 1,
    "flux_schnell_max": 2,
    "flux_dev_min": 3,
    "flux_dev_max": 5,
    "sdxl_hq_min": 5,
    "sdxl_hq_max": 8,
    "midjourney_style_min": 8,
    "midjourney_style_max": 15,
    "video_min": 20,
    "video_max": 100,

    # Voice. Voice is capped at 2 minutes in Telegram handler.
    "voice_0_15_sec": 1,
    "voice_16_60_sec": 2,
    "voice_61_120_sec": 4,
    "voice_default": 2,

    # Heavy work
    "large_file_default": 3,
    "heavy_reasoning_default": 5,
}

# Hidden fair-use defaults. Not shown as hard marketing promises.
HIDDEN_FAIR_USE = {
    "image_per_hour_free": 3,
    "image_per_hour_plus": 12,
    "image_per_hour_pro": 30,
    "voice_per_hour_plus": 20,
    "voice_per_hour_pro": 50,
    "heavy_prompt_cooldown_seconds": 20,
    "max_prompt_chars_free": 2500,
    "max_prompt_chars_plus": 7000,
    "max_prompt_chars_pro": 14000,
}

# Every invited friend gives +10 VS tokens immediately.
# Milestones below are additional big bonuses.
REFERRAL_FRIEND_BONUS_TOKENS = 10

REFERRAL_REWARDS: dict[int, dict[str, Any]] = {
    100: {
        "type": "combo",
        "tokens": 1200,
        "plan": "PLUS",
        "days": 10,
        "title": "+1200 VS токенов и +10 дней PLUS",
    },
    500: {
        "type": "combo",
        "tokens": 7000,
        "plan": "PLUS",
        "days": 30,
        "title": "+7000 VS токенов и +30 дней PLUS",
    },
    1000: {
        "type": "combo",
        "tokens": 16000,
        "plan": "PRO",
        "days": 30,
        "title": "+16000 VS токенов и +30 дней PRO",
    },
}

PLAN_LEVELS = {
    "FREE": 0,
    "PLUS": 1,
    "PRO": 2,
}

MODEL_REQUIRED_PLAN = {
    "gpt": "FREE",
    "gemini": "FREE",
    "claude": "FREE",
    "nanobanana": "FREE",
    "gptimage": "FREE",
}


def plan_level(plan: str | None) -> int:
    return PLAN_LEVELS.get((plan or "FREE").upper(), 0)


def model_required_plan(model: str) -> str:
    return MODEL_REQUIRED_PLAN.get(model, "FREE")


def has_model_access(user_plan: str | None, model: str) -> bool:
    return plan_level(user_plan) >= plan_level(model_required_plan(model))


def is_paid_plan(plan: str | None) -> bool:
    return (plan or "FREE").upper() in {"PLUS", "PRO"}


def has_active_paid_subscription(user: Any) -> bool:
    plan = (user["plan"] if hasattr(user, "__getitem__") else None) or "FREE"
    plan_until = user["plan_until"] if hasattr(user, "__getitem__") and "plan_until" in dict(user) else None
    return is_paid_plan(plan) and bool(plan_until) and plan_until > datetime.utcnow()


def subscription_bonus_tokens(plan: str, months: int) -> int:
    # Bonus is granted once per successful subscription purchase.
    tariff = TARIFFS.get((plan or "").upper())
    if not tariff:
        return 0
    return int(tariff.get("bonus_tokens") or 0)


def tariff_button_text(plan: str) -> str:
    tariff = TARIFFS[plan]
    return f"{tariff['title']} — до {tariff['daily_limit']} запросов / день"


def period_button_text(plan: str, months: int) -> str:
    tariff = TARIFFS[plan]
    stars = tariff["prices"][months]
    rub = tariff["rub_prices"][months]
    return f"{months} мес. — ⭐ {stars} / {rub} ₽"


def token_pack_button_text(amount: int) -> str:
    pack = VS_TOKEN_PACKS[amount]
    return f"💵 {amount} VS токенов — ⭐ {pack['stars']} / {pack['rub']} ₽"


def image_token_cost(model: str, is_edit: bool = False) -> int:
    model = (model or "").lower()
    if model == "nanobanana":
        return VS_TOKEN_COSTS["nanobanana_edit" if is_edit else "nanobanana_generate"]
    if model == "gptimage":
        return VS_TOKEN_COSTS["gptimage_edit" if is_edit else "gptimage_generate"]
    return VS_TOKEN_COSTS["image_edit_default" if is_edit else "image_default"]


def image_cost_text(model: str, is_edit: bool = False) -> str:
    return f"Стоимость: {image_token_cost(model, is_edit)} VS токенов"


def voice_token_cost(duration_seconds: int | None) -> int:
    duration = int(duration_seconds or 0)
    if duration <= 15:
        return VS_TOKEN_COSTS["voice_0_15_sec"]
    if duration <= 60:
        return VS_TOKEN_COSTS["voice_16_60_sec"]
    return VS_TOKEN_COSTS["voice_61_120_sec"]


def voice_cost_text(duration_seconds: int | None) -> str:
    duration = int(duration_seconds or 0)
    return f"Стоимость голосового: {voice_token_cost(duration)} VS токенов"


def referral_reward_title(milestone: int, reward_type: str | None = None, reward_value: str | None = None) -> str:
    reward = REFERRAL_REWARDS.get(int(milestone))
    if reward:
        return reward["title"]

    if reward_type in {"vs_tokens", "tokens"} and reward_value:
        return f"+{reward_value} VS токенов"

    if reward_type == "combo":
        return "бонус за партнёрский порог"

    if reward_type == "requests" and reward_value:
        return f"+{reward_value} запросов"

    if reward_type == "plan_days" and reward_value and ":" in reward_value:
        plan, days = reward_value.split(":", 1)
        return f"+{days} дней {plan}"

    return "бонус"


def premium_text() -> str:
    return """💳 Купить подписку

❤️‍🔥 FREE
• ChatGPT — GPT-5 mini
• Gemini — 3.1 Flash
• Claude — Sonnet 4.6
• Nano Banana Pro
• GPT Image 2

• 14 запросов в день
• из них 3 Image
• 98 запросов в неделю

⭐ PLUS / до 100 запросов в день
• последние AI-модели
• работа с документами
• голосовые ответы
• разумное использование
• без тяжёлых моделей глубокого анализа
💵 +100 VS токенов при покупке

💎 PRO / до 200 запросов в день
• всё из PLUS
• GPT-5 чат
• Claude
• память
• файлы
• приоритет
• длинный контекст и кодинг
💵 +300 VS токенов при покупке

💵 VS токены покупаются отдельно и тратятся на изображения, редактирование фото, видео AI, голос, большие документы и премиум-генерацию.

Важно: 💵 VS токены можно использовать только при активной подписке."""

