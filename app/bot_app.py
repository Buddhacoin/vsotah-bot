import asyncio
import base64
import os
import time
import traceback
from datetime import date, datetime, timedelta
from io import BytesIO

import asyncpg
from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    CallbackQuery,
    PreCheckoutQuery,
    LabeledPrice,
    BotCommand,
    BotCommandScopeDefault,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllChatAdministrators,
    BotCommandScopeChat,
    LinkPreviewOptions,
    BufferedInputFile,
)

from app.ai.router import (
    ai_router,
    research_router,
    business_router,
    code_router,
    web_router,
    web_is_configured,
    search_web,
    vision_router,
    generate_nano_banana_image,
    generate_gpt_image,
    edit_gpt_image,
)
from app.ai.file_router import (
    extract_document_text,
    file_router,
    analyze_pdf_images_with_openai,
    build_file_status_text,
)
from app.ai.voice_router import transcribe_voice, text_to_speech, build_voice_user_message
from app.referrals import build_referral_link, parse_referral_code, build_invite_text, build_telegram_share_url
    
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5.4-mini")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.4-mini")
ANTHROPIC_TEXT_MODEL = os.getenv("ANTHROPIC_TEXT_MODEL", "claude-sonnet-4-6")
ANTHROPIC_VISION_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-6")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
NANO_BANANA_MODEL = os.getenv("NANO_BANANA_MODEL", "imagen-4.0-generate-001")
GPT_IMAGE_MODEL = os.getenv("GPT_IMAGE_MODEL", "gpt-image-1")
GPT_IMAGE_QUALITY = os.getenv("GPT_IMAGE_QUALITY", "high")

# Speed settings. You can override these in Railway Variables if needed.
TEXT_HISTORY_LIMIT = int(os.getenv("TEXT_HISTORY_LIMIT", "6"))
VISION_HISTORY_LIMIT = int(os.getenv("VISION_HISTORY_LIMIT", "2"))
TEXT_MAX_TOKENS = int(os.getenv("TEXT_MAX_TOKENS", "1200"))
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "900"))
AI_TIMEOUT_SECONDS = int(os.getenv("AI_TIMEOUT_SECONDS", "75"))
MAX_DOCUMENT_SIZE = int(os.getenv("MAX_DOCUMENT_SIZE", str(10 * 1024 * 1024)))
MAX_VOICE_SIZE = int(os.getenv("MAX_VOICE_SIZE", str(20 * 1024 * 1024)))
FILE_TEXT_LIMIT = int(os.getenv("FILE_TEXT_LIMIT", "18000"))
VOICE_REPLY_ENABLED = os.getenv("VOICE_REPLY_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
ADMIN_ERROR_NOTIFICATIONS = os.getenv("ADMIN_ERROR_NOTIFICATIONS", "false").lower() in {"1", "true", "yes", "on"}

PORT = int(os.getenv("PORT", "8080"))
STARTED_AT = datetime.utcnow()

ADMIN_IDS = {
    int(x.strip())
    for x in os.getenv("ADMIN_IDS", "").split(",")
    if x.strip().isdigit()
}

FREE_DAILY_LIMIT = 15
FREE_WEEKLY_LIMIT = 105
FREE_DAILY_IMAGE_LIMIT = 5

PLAN_WEEKLY_LIMITS = {
    "FREE": 105,
    "PLUS": 500,
    "PRO": 1400,
    "VIP": None,
}

TARIFFS = {
    "PLUS": {
        "title": "PLUS",
        "description": "500 запросов в неделю",
        "prices": {1: 199, 3: 400, 6: 800, 12: 1600},
    },
    "PRO": {
        "title": "PRO",
        "description": "1400 запросов в неделю",
        "prices": {1: 499, 3: 1000, 6: 2000, 12: 3000},
    },
    "VIP": {
        "title": "VIP",
        "description": "Безлимит",
        "prices": {1: 1499, 3: 3000, 6: 6000, 12: 9900},
    },
}

REFERRAL_REWARDS = {
    1: {"type": "requests", "amount": 50, "title": "+50 запросов"},
    3: {"type": "plan_days", "plan": "PLUS", "days": 3, "title": "+3 дня PLUS"},
    10: {"type": "plan_days", "plan": "PRO", "days": 5, "title": "+5 дней PRO"},
    25: {"type": "plan_days", "plan": "VIP", "days": 7, "title": "VIP статус на 7 дней"},
    100: {"type": "plan_days", "plan": "VIP", "days": 30, "title": "VIP статус на 30 дней"},
}


def referral_reward_title(milestone: int, reward_type: str | None = None, reward_value: str | None = None) -> str:
    reward = REFERRAL_REWARDS.get(int(milestone))
    if reward:
        return reward["title"]

    if reward_type == "requests" and reward_value:
        return f"+{reward_value} запросов"

    if reward_type == "plan_days" and reward_value and ":" in reward_value:
        plan, days = reward_value.split(":", 1)
        return f"+{days} дней {plan}"

    return "бонус"

SPAM_WINDOW_SECONDS = 8
SPAM_MAX_MESSAGES = 5
SPAM_BLOCK_SECONDS = 60

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


db_pool = None
recent_starts = {}
BOT_USERNAME_CACHE = None


async def get_bot_username(bot_obj: Bot) -> str:
    global BOT_USERNAME_CACHE
    if BOT_USERNAME_CACHE:
        return BOT_USERNAME_CACHE
    bot_info = await bot_obj.get_me()
    BOT_USERNAME_CACHE = bot_info.username
    return BOT_USERNAME_CACHE


def no_preview():
    return LinkPreviewOptions(is_disabled=True)


def main_menu():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👤 Профиль", callback_data="profile"),
                InlineKeyboardButton(text="💳 Купить подписку", callback_data="premium"),
            ],
            [
                InlineKeyboardButton(text="🤖 Выбрать AI", callback_data="models"),
                InlineKeyboardButton(text="💰 Заработать", callback_data="earn"),
            ],
            [InlineKeyboardButton(text="🧠 Наши каналы", callback_data="channels")],
        ]
    )


def models_menu():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🌀 ChatGPT — GPT-4o mini", callback_data="set_model_gpt")],
            [InlineKeyboardButton(text="✦ Gemini — 2.5 Flash", callback_data="set_model_gemini")],
            [InlineKeyboardButton(text="✴️ Claude — Sonnet", callback_data="set_model_claude")],
            [InlineKeyboardButton(text="🍌 Nano Banana Pro", callback_data="set_model_nanobanana")],
            [InlineKeyboardButton(text="🌀 Sora GPT Image", callback_data="set_model_gptimage")],
            [InlineKeyboardButton(text="← Назад", callback_data="back_main")],
        ]
    )


def tariffs_menu():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⭐ PLUS — 500 запросов / неделя", callback_data="tariff_PLUS")],
            [InlineKeyboardButton(text="💎 PRO — 1400 запросов / неделя", callback_data="tariff_PRO")],
            [InlineKeyboardButton(text="👑 VIP — безлимит", callback_data="tariff_VIP")],
            [InlineKeyboardButton(text="← Назад", callback_data="back_main")],
        ]
    )


def channels_menu():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⚡ Молния Live", url="https://t.me/MolniyaLiveNews")],
            [InlineKeyboardButton(text="⚡ VSotah AI", url="https://t.me/VSotahAI")],
            [InlineKeyboardButton(text="← Назад", callback_data="back_main")],
        ]
    )



def referral_menu(bot_username: str, user_id: int):
    referral_link = build_referral_link(bot_username, user_id)
    share_url = build_telegram_share_url(referral_link)
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📨 Пригласить друзей", callback_data="earn_invite")],
            [InlineKeyboardButton(text="🏆 Топ партнёров", callback_data="earn_top")],
            [InlineKeyboardButton(text="📊 Моя статистика", callback_data="earn_stats")],
            [InlineKeyboardButton(text="📤 Поделиться в Telegram", url=share_url)],
            [InlineKeyboardButton(text="← Назад", callback_data="back_main")],
        ]
    )


def referral_back_menu(bot_username: str, user_id: int):
    referral_link = build_referral_link(bot_username, user_id)
    share_url = build_telegram_share_url(referral_link)
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📤 Поделиться в Telegram", url=share_url)],
            [InlineKeyboardButton(text="← В партнёрку", callback_data="earn")],
            [InlineKeyboardButton(text="← Главное меню", callback_data="back_main")],
        ]
    )

def period_menu(plan: str):
    prices = TARIFFS[plan]["prices"]
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=f"1 месяц — ⭐ {prices[1]}", callback_data=f"period_{plan}_1")],
            [InlineKeyboardButton(text=f"3 месяца — ⭐ {prices[3]}", callback_data=f"period_{plan}_3")],
            [InlineKeyboardButton(text=f"6 месяцев — ⭐ {prices[6]}", callback_data=f"period_{plan}_6")],
            [InlineKeyboardButton(text=f"12 месяцев — ⭐ {prices[12]}", callback_data=f"period_{plan}_12")],
            [InlineKeyboardButton(text="← Назад", callback_data="premium")],
        ]
    )


def payment_method_menu(plan: str, months: int):
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="💳 Карта / СБП — скоро", callback_data="rub_disabled")],
            [InlineKeyboardButton(text="⭐ Telegram Stars", callback_data=f"pay_stars_{plan}_{months}")],
            [InlineKeyboardButton(text="← Назад", callback_data=f"tariff_{plan}")],
        ]
    )


def get_week_start():
    today = date.today()
    return today - timedelta(days=today.weekday())


def add_months_rough(months: int):
    return datetime.utcnow() + timedelta(days=30 * months)


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


PLAN_LEVELS = {
    "FREE": 0,
    "PLUS": 1,
    "PRO": 2,
    "VIP": 3,
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


def model_display_name(model: str) -> str:
    names = {
        "gpt": "🌀 ChatGPT — GPT-4o mini",
        "gemini": "✦ Gemini — 2.5 Flash",
        "claude": "✴️ Claude — Sonnet",
        "nanobanana": "🍌 Nano Banana Pro",
        "gptimage": "🌀 Sora GPT Image",
    }
    return names.get(model, "🌀 ChatGPT — GPT-4o mini")


def premium_required_text(model: str) -> str:
    required = model_required_plan(model)
    return (
        f"🔒 {model_display_name(model)} доступна с тарифа {required}.\n\n"
        "Выберите подходящий тариф, чтобы открыть более мощные нейросети."
    )


def welcome_text():
    return """👋 Добро пожаловать в @VSotahBot

Ваш AI-бот для работы с нейросетями в одном месте.

📝 Генерация текста:
• ChatGPT
• Claude
• Gemini

🌇 Генерация изображений:
• Nano Banana Pro
• Sora GPT Image

📷 Анализ фото:
• вопросы по изображениям
• распознавание текста
• помощь с заданиями по фото

🧠 Наши каналы:
• Наш канал: <a href='https://t.me/MolniyaLiveNews'>Молния Live</a>
• Канал support: <a href='https://t.me/VSotahAI'>VSotah AI</a>

Напишите вопрос, отправьте фото или выберите действие ниже."""


def premium_text():
    return """💳 Купить подписку

🟢 FREE
• ChatGPT — GPT-4o mini
• Gemini — 2.5 Flash
• Claude — Sonnet
• Nano Banana Pro
• Sora GPT Image

• 15 запросов в день
• из них 5 Image
• 105 запросов в неделю

⭐ PLUS — 500 запросов в неделю
• Все модели
• больше лимитов

💎 PRO — 1400 запросов в неделю
• Все модели
• больше лимитов

👑 VIP — безлимит
• Все модели
• максимум возможностей"""

def channels_text():
    return """🧠 Наши каналы:

• Наш канал: <a href='https://t.me/MolniyaLiveNews'>Молния Live</a>
• Канал support: <a href='https://t.me/VSotahAI'>VSotah AI</a>"""


async def init_db():
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL)

    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                telegram_id BIGINT PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                plan TEXT DEFAULT 'FREE',
                selected_model TEXT DEFAULT 'gpt',
                daily_used INTEGER DEFAULT 0,
                weekly_used INTEGER DEFAULT 0,
                daily_image_used INTEGER DEFAULT 0,
                day_start DATE DEFAULT CURRENT_DATE,
                week_start DATE DEFAULT CURRENT_DATE,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS plan_until TIMESTAMP;")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS selected_model TEXT DEFAULT 'gpt';")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS daily_used INTEGER DEFAULT 0;")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS weekly_used INTEGER DEFAULT 0;")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS daily_image_used INTEGER DEFAULT 0;")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS bonus_requests INTEGER DEFAULT 0;")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS day_start DATE DEFAULT CURRENT_DATE;")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS week_start DATE DEFAULT CURRENT_DATE;")

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                telegram_id BIGINT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS payments (
                id SERIAL PRIMARY KEY,
                telegram_id BIGINT NOT NULL,
                plan TEXT NOT NULL,
                amount INTEGER NOT NULL,
                currency TEXT NOT NULL,
                payload TEXT NOT NULL,
                telegram_payment_charge_id TEXT,
                provider_payment_charge_id TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        await conn.execute("ALTER TABLE payments ADD COLUMN IF NOT EXISTS months INTEGER DEFAULT 1;")

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS spam_state (
                telegram_id BIGINT PRIMARY KEY,
                window_start BIGINT DEFAULT 0,
                message_count INTEGER DEFAULT 0,
                blocked_until BIGINT DEFAULT 0
            );
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                telegram_id BIGINT,
                event_type TEXT NOT NULL,
                details TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS referrals (
                id SERIAL PRIMARY KEY,
                referrer_id BIGINT NOT NULL,
                referred_id BIGINT NOT NULL UNIQUE,
                status TEXT DEFAULT 'joined',
                reward_status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT NOW(),
                paid_at TIMESTAMP NULL
            );
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_referrals_referrer_id ON referrals(referrer_id);")
        await conn.execute("ALTER TABLE referrals ALTER COLUMN status SET DEFAULT 'joined';")
        await conn.execute("ALTER TABLE referrals ADD COLUMN IF NOT EXISTS reward_status TEXT DEFAULT 'pending';")
        await conn.execute("ALTER TABLE referrals ADD COLUMN IF NOT EXISTS paid_at TIMESTAMP NULL;")
        await conn.execute("ALTER TABLE referrals ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'start_link';")
        await conn.execute("ALTER TABLE referrals ADD COLUMN IF NOT EXISTS abuse_flags TEXT DEFAULT '';")

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS referral_rewards (
                id SERIAL PRIMARY KEY,
                referrer_id BIGINT NOT NULL,
                milestone INTEGER NOT NULL,
                reward_type TEXT NOT NULL,
                reward_value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(referrer_id, milestone)
            );
        """)
        await conn.execute("ALTER TABLE referral_rewards ADD COLUMN IF NOT EXISTS reward_title TEXT;")
        await conn.execute("ALTER TABLE referral_rewards ADD COLUMN IF NOT EXISTS source_count INTEGER DEFAULT 0;")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_referral_rewards_referrer_id ON referral_rewards(referrer_id);")

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS referral_abuse_events (
                id SERIAL PRIMARY KEY,
                referrer_id BIGINT,
                referred_id BIGINT,
                event_type TEXT NOT NULL,
                details TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_referral_abuse_referrer_id ON referral_abuse_events(referrer_id);")


async def log_event(telegram_id: int | None, event_type: str, details: str = ""):
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO events (telegram_id, event_type, details) VALUES ($1, $2, $3)",
                telegram_id,
                event_type,
                details[:1000],
            )
    except Exception as e:
        print(f"LOG EVENT ERROR: {e}")


async def track_referral_start(referred_id: int, referrer_id: int | None):
    if not referrer_id:
        return False

    if referrer_id == referred_id:
        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO referral_abuse_events (referrer_id, referred_id, event_type, details) VALUES ($1, $2, $3, $4)",
                    referrer_id,
                    referred_id,
                    "self_ref",
                    "user tried to invite himself",
                )
            await log_event(referred_id, "referral_self_ref", str(referrer_id))
        except Exception:
            pass
        return False

    try:
        async with db_pool.acquire() as conn:
            existing_user = await conn.fetchrow("SELECT telegram_id FROM users WHERE telegram_id=$1", referred_id)
            if existing_user:
                await conn.execute(
                    "INSERT INTO referral_abuse_events (referrer_id, referred_id, event_type, details) VALUES ($1, $2, $3, $4)",
                    referrer_id,
                    referred_id,
                    "existing_user",
                    "start link opened by existing user",
                )
                return False

            inserted = await conn.fetchrow("""
                INSERT INTO referrals (referrer_id, referred_id, status, source)
                VALUES ($1, $2, 'joined', 'start_link')
                ON CONFLICT (referred_id) DO NOTHING
                RETURNING id
            """, referrer_id, referred_id)

            if not inserted:
                await conn.execute(
                    "INSERT INTO referral_abuse_events (referrer_id, referred_id, event_type, details) VALUES ($1, $2, $3, $4)",
                    referrer_id,
                    referred_id,
                    "duplicate_referral",
                    "referred user already has referrer",
                )
                return False

        await log_event(referred_id, "referral_joined", str(referrer_id))
        await process_referral_rewards(referrer_id)
        return True
    except Exception as e:
        print(f"REFERRAL TRACK ERROR: {short_error_text(e)}")
        return False

async def apply_plan_days_bonus(telegram_id: int, bonus_plan: str, days: int):
    bonus_plan = (bonus_plan or "PLUS").upper()
    now = datetime.utcnow()

    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT plan, plan_until FROM users WHERE telegram_id=$1", telegram_id)
        if not user:
            return

        current_plan = (user["plan"] or "FREE").upper()
        current_until = user["plan_until"]

        if current_plan != "FREE" and current_until and current_until > now:
            base_until = current_until
        else:
            base_until = now

        final_plan = bonus_plan
        if plan_level(current_plan) > plan_level(bonus_plan) and current_until and current_until > now:
            final_plan = current_plan

        await conn.execute("""
            UPDATE users
            SET plan=$2, plan_until=$3
            WHERE telegram_id=$1
        """, telegram_id, final_plan, base_until + timedelta(days=days))


async def grant_referral_reward(referrer_id: int, milestone: int, reward: dict, source_count: int) -> bool:
    reward_type = reward["type"]
    if reward_type == "requests":
        reward_value = str(reward["amount"])
    else:
        reward_value = f"{reward['plan']}:{reward['days']}"

    reward_title = reward.get("title") or referral_reward_title(milestone, reward_type, reward_value)

    async with db_pool.acquire() as conn:
        inserted = await conn.fetchrow("""
            INSERT INTO referral_rewards (referrer_id, milestone, reward_type, reward_value, reward_title, source_count)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (referrer_id, milestone) DO NOTHING
            RETURNING id
        """, referrer_id, milestone, reward_type, reward_value, reward_title, source_count)

        if not inserted:
            return False

        if reward_type == "requests":
            await conn.execute("""
                UPDATE users
                SET bonus_requests = COALESCE(bonus_requests, 0) + $2
                WHERE telegram_id=$1
            """, referrer_id, int(reward["amount"]))

        await conn.execute("""
            UPDATE referrals
            SET reward_status='rewarded'
            WHERE referrer_id=$1
              AND id IN (
                  SELECT id FROM referrals
                  WHERE referrer_id=$1
                  ORDER BY created_at
                  LIMIT $2
              )
        """, referrer_id, milestone)

    if reward_type == "plan_days":
        await apply_plan_days_bonus(referrer_id, reward["plan"], int(reward["days"]))

    await log_event(referrer_id, "referral_reward", f"{milestone}: {reward_title}")
    return True


async def process_referral_rewards(referrer_id: int) -> list[str]:
    if not referrer_id:
        return []

    granted: list[str] = []
    try:
        async with db_pool.acquire() as conn:
            invited_count = await conn.fetchval(
                "SELECT COUNT(*) FROM referrals WHERE referrer_id=$1",
                referrer_id,
            )

        invited_count = invited_count or 0
        for milestone, reward in sorted(REFERRAL_REWARDS.items()):
            if invited_count >= milestone:
                was_granted = await grant_referral_reward(referrer_id, milestone, reward, invited_count)
                if was_granted:
                    granted.append(reward["title"])

    except Exception as e:
        print(f"REFERRAL REWARD ERROR: {short_error_text(e)}")

    return granted


async def mark_referral_paid(referred_id: int, plan: str) -> int | None:
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("""
                UPDATE referrals
                SET status='paid', paid_at=COALESCE(paid_at, NOW())
                WHERE referred_id=$1 AND status <> 'paid'
                RETURNING referrer_id
            """, referred_id)

        if not row:
            return None

        referrer_id = row["referrer_id"]
        await log_event(referred_id, "referral_paid", f"{plan} from {referrer_id}")
        await process_referral_rewards(referrer_id)
        return referrer_id
    except Exception as e:
        print(f"REFERRAL PAID ERROR: {short_error_text(e)}")
        return None


async def get_referral_stats(telegram_id: int) -> dict:
    try:
        async with db_pool.acquire() as conn:
            invited = await conn.fetchval("SELECT COUNT(*) FROM referrals WHERE referrer_id=$1", telegram_id)
            paid = await conn.fetchval("SELECT COUNT(*) FROM referrals WHERE referrer_id=$1 AND status='paid'", telegram_id)
            rewards_count = await conn.fetchval("SELECT COUNT(*) FROM referral_rewards WHERE referrer_id=$1", telegram_id)
            bonus_requests = await conn.fetchval("SELECT COALESCE(bonus_requests, 0) FROM users WHERE telegram_id=$1", telegram_id)
            rewards = await conn.fetch("""
                SELECT milestone, reward_type, reward_value, COALESCE(reward_title, '') AS reward_title, created_at
                FROM referral_rewards
                WHERE referrer_id=$1
                ORDER BY milestone
            """, telegram_id)
        return {
            "invited": invited or 0,
            "paid": paid or 0,
            "pending_rewards": 0,
            "rewards_count": rewards_count or 0,
            "bonus_requests": bonus_requests or 0,
            "rewards": rewards or [],
        }
    except Exception as e:
        print(f"REFERRAL STATS ERROR: {short_error_text(e)}")
        return {"invited": 0, "paid": 0, "pending_rewards": 0, "rewards_count": 0, "bonus_requests": 0, "rewards": []}


def build_reward_history_lines(rewards) -> str:
    if not rewards:
        return "• Пока бонусов нет"

    lines = []
    for row in rewards:
        milestone = int(row["milestone"])
        title = row["reward_title"] or referral_reward_title(milestone, row["reward_type"], row["reward_value"])
        created_at = row["created_at"].strftime("%d.%m.%Y") if row["created_at"] else ""
        suffix = f" — {created_at}" if created_at else ""
        lines.append(f"• {milestone} друг → {title}{suffix}")
    return "\n".join(lines)


async def build_referral_text(bot_obj: Bot, user_id: int) -> str:
    bot_username = await get_bot_username(bot_obj)
    referral_link = build_referral_link(bot_username, user_id)
    stats = await get_referral_stats(user_id)

    return (
        "💰 Заработать с VSotah AI\n\n"
        "Приглашай друзей по своей ссылке и получай бонусы внутри бота.\n\n"
        f"🔗 Твоя ссылка:\n{referral_link}\n\n"
        "📊 Твоя статистика:\n"
        f"• Приглашено: {stats['invited']}\n"
        f"• Купили подписку: {stats['paid']}\n"
        f"• Бонусов получено: {stats['rewards_count']}\n\n"
        "🎁 Бонусы за приглашения:\n"
        "• 1 друг → +50 запросов\n"
        "• 3 друга → +3 дня PLUS\n"
        "• 10 друзей → +5 дней PRO\n"
        "• 25 друзей → VIP статус 7 дней\n"
        "• 100 друзей → VIP статус 30 дней\n\n"
        "Бонусы выдаются автоматически внутри бота."
    )


async def build_referral_invite_text(bot_obj: Bot, user_id: int) -> str:
    bot_username = await get_bot_username(bot_obj)
    referral_link = build_referral_link(bot_username, user_id)
    invite_text = build_invite_text(referral_link)
    return (
        "📨 Пригласить друзей\n\n"
        "Нажмите кнопку «Поделиться в Telegram» или просто скопируйте текст ниже.\n\n"
        f"{invite_text}"
    )


async def build_referral_stats_text(user_id: int) -> str:
    stats = await get_referral_stats(user_id)
    rewards_text = build_reward_history_lines(stats.get("rewards", []))
    return (
        "📊 Моя статистика партнёрки\n\n"
        f"• Приглашено всего: {stats['invited']}\n"
        f"• Купили подписку: {stats['paid']}\n"
        f"• Бонусных запросов сейчас: {stats['bonus_requests']}\n"
        f"• Бонусов получено: {stats['rewards_count']}\n\n"
        "🎁 Уже получено:\n"
        f"{rewards_text}\n\n"
        "Бонусы начисляются автоматически за новых друзей, которые впервые зашли по вашей ссылке."
    )

async def build_referral_bonuses_text(user_id: int) -> str:
    stats = await get_referral_stats(user_id)
    return (
        "🎁 Бонусы\n\n"
        f"Сейчас приглашено: {stats['invited']}\n"
        f"Бонусов получено: {stats['rewards_count']}"
    )


async def build_referral_leaderboard_text() -> str:
    async with db_pool.acquire() as conn:
        invited_rows = await conn.fetch("""
            SELECT r.referrer_id, COALESCE(u.username, u.first_name, r.referrer_id::text) AS name, COUNT(*) AS invited
            FROM referrals r
            LEFT JOIN users u ON u.telegram_id = r.referrer_id
            GROUP BY r.referrer_id, u.username, u.first_name
            ORDER BY invited DESC, r.referrer_id
            LIMIT 10
        """)
        paid_rows = await conn.fetch("""
            SELECT r.referrer_id, COALESCE(u.username, u.first_name, r.referrer_id::text) AS name, COUNT(*) AS paid
            FROM referrals r
            LEFT JOIN users u ON u.telegram_id = r.referrer_id
            WHERE r.status='paid'
            GROUP BY r.referrer_id, u.username, u.first_name
            ORDER BY paid DESC, r.referrer_id
            LIMIT 10
        """)

    def clean_name(value):
        value = str(value or "партнёр")
        if value.isdigit():
            return f"ID {value}"
        if not value.startswith("@") and " " not in value:
            return f"@{value}"
        return value

    invited_text = "Пока нет приглашений. Станьте первым партнёром 🚀"
    if invited_rows:
        invited_text = "\n".join(
            f"{index}. {clean_name(row['name'])} — {row['invited']}"
            for index, row in enumerate(invited_rows, start=1)
        )

    paid_text = "Пока нет оплат по партнёрке."
    if paid_rows:
        paid_text = "\n".join(
            f"{index}. {clean_name(row['name'])} — {row['paid']}"
            for index, row in enumerate(paid_rows, start=1)
        )

    return (
        "🏆 Топ партнёров\n\n"
        "👥 По приглашениям:\n"
        f"{invited_text}\n\n"
        "💳 По покупкам подписки:\n"
        f"{paid_text}"
    )

async def setup_bot_info():
    # /start handler remains active, but /start must NOT be shown in Telegram's command menu.
    # Telegram clients cache commands aggressively, so we clear and set several scopes/languages.
    commands = [
        BotCommand(command="account", description="👤 Мой профиль"),
        BotCommand(command="premium", description="💳 Купить подписку"),
        BotCommand(command="models", description="🤖 Выбрать AI"),
        BotCommand(command="research", description="🔎 Deep Research"),
        BotCommand(command="web", description="🌐 Live Web AI"),
        BotCommand(command="business", description="💼 Business AI"),
        BotCommand(command="code", description="💻 Code AI"),
        BotCommand(command="referral", description="💰 Заработать"),
        BotCommand(command="channels", description="🧠 Наши каналы"),
        BotCommand(command="deletecontext", description="💬 Удалить контекст"),
    ]

    scopes_to_clear = [
        BotCommandScopeDefault(),
        BotCommandScopeAllPrivateChats(),
        BotCommandScopeAllGroupChats(),
        BotCommandScopeAllChatAdministrators(),
    ]
    language_codes = [None, "ru", "en"]

    for scope in scopes_to_clear:
        for language_code in language_codes:
            try:
                kwargs = {"scope": scope}
                if language_code:
                    kwargs["language_code"] = language_code
                await bot.delete_my_commands(**kwargs)
            except Exception as e:
                print(f"DELETE COMMANDS WARNING: {short_error_text(e)}")

    for scope in [BotCommandScopeDefault(), BotCommandScopeAllPrivateChats()]:
        for language_code in language_codes:
            try:
                kwargs = {"scope": scope}
                if language_code:
                    kwargs["language_code"] = language_code
                await bot.set_my_commands(commands, **kwargs)
            except Exception as e:
                print(f"SET COMMANDS WARNING: {short_error_text(e)}")



async def clear_start_for_chat(chat_id: int):
    """Telegram clients cache command menus. Set a chat-specific command list
    without /start so the annoying /start suggestion disappears after the user
    opens the bot. The /start handler itself continues to work.
    """
    commands = [
        BotCommand(command="account", description="👤 Мой профиль"),
        BotCommand(command="premium", description="💳 Купить подписку"),
        BotCommand(command="models", description="🤖 Выбрать AI"),
        BotCommand(command="research", description="🔎 Deep Research"),
        BotCommand(command="web", description="🌐 Live Web AI"),
        BotCommand(command="business", description="💼 Business AI"),
        BotCommand(command="code", description="💻 Code AI"),
        BotCommand(command="referral", description="💰 Заработать"),
        BotCommand(command="channels", description="🧠 Наши каналы"),
        BotCommand(command="deletecontext", description="💬 Удалить контекст"),
    ]
    try:
        await bot.delete_my_commands(scope=BotCommandScopeChat(chat_id=chat_id))
        await bot.set_my_commands(commands, scope=BotCommandScopeChat(chat_id=chat_id))
    except Exception as e:
        print(f"CHAT COMMANDS WARNING: {short_error_text(e)}")


async def get_or_create_user_by_data(telegram_id, username=None, first_name=None):
    today = date.today()
    week_start = get_week_start()

    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM users WHERE telegram_id=$1", telegram_id)

        if not user:
            await conn.execute("""
                INSERT INTO users (telegram_id, username, first_name, day_start, week_start)
                VALUES ($1, $2, $3, $4, $5)
            """, telegram_id, username, first_name, today, week_start)
            await log_event(telegram_id, "new_user", username or "")
        else:
            await conn.execute("UPDATE users SET username=$2, first_name=$3 WHERE telegram_id=$1", telegram_id, username, first_name)

        user = await conn.fetchrow("SELECT * FROM users WHERE telegram_id=$1", telegram_id)

        if user["day_start"] != today:
            await conn.execute(
                "UPDATE users SET daily_used=0, daily_image_used=0, day_start=$2 WHERE telegram_id=$1",
                telegram_id,
                today,
            )

        if user["week_start"] != week_start:
            await conn.execute("UPDATE users SET weekly_used=0, week_start=$2 WHERE telegram_id=$1", telegram_id, week_start)

        user = await conn.fetchrow("SELECT * FROM users WHERE telegram_id=$1", telegram_id)

        if user["plan"] != "FREE" and user["plan_until"] and user["plan_until"] < datetime.utcnow():
            await conn.execute("UPDATE users SET plan='FREE', plan_until=NULL WHERE telegram_id=$1", telegram_id)
            await log_event(telegram_id, "plan_expired", user["plan"])

        return await conn.fetchrow("SELECT * FROM users WHERE telegram_id=$1", telegram_id)


async def get_or_create_user(message: Message):
    return await get_or_create_user_by_data(
        telegram_id=message.from_user.id,
        username=message.from_user.username,
        first_name=message.from_user.first_name,
    )


async def check_spam(telegram_id: int):
    now = int(time.time())

    async with db_pool.acquire() as conn:
        state = await conn.fetchrow("SELECT * FROM spam_state WHERE telegram_id=$1", telegram_id)

        if not state:
            await conn.execute("""
                INSERT INTO spam_state (telegram_id, window_start, message_count, blocked_until)
                VALUES ($1, $2, 1, 0)
            """, telegram_id, now)
            return True, None

        if state["blocked_until"] and state["blocked_until"] > now:
            return False, state["blocked_until"] - now

        if now - state["window_start"] > SPAM_WINDOW_SECONDS:
            await conn.execute("UPDATE spam_state SET window_start=$2, message_count=1, blocked_until=0 WHERE telegram_id=$1", telegram_id, now)
            return True, None

        new_count = state["message_count"] + 1

        if new_count > SPAM_MAX_MESSAGES:
            blocked_until = now + SPAM_BLOCK_SECONDS
            await conn.execute("UPDATE spam_state SET message_count=$2, blocked_until=$3 WHERE telegram_id=$1", telegram_id, new_count, blocked_until)
            await log_event(telegram_id, "spam_block", str(SPAM_BLOCK_SECONDS))
            return False, SPAM_BLOCK_SECONDS

        await conn.execute("UPDATE spam_state SET message_count=$2 WHERE telegram_id=$1", telegram_id, new_count)

    return True, None


async def check_limit(user):
    plan = user["plan"]
    bonus_requests = dict(user).get("bonus_requests") or 0

    if plan == "VIP":
        return True, None

    if bonus_requests > 0:
        return True, None

    if plan == "FREE":
        if user["daily_used"] >= FREE_DAILY_LIMIT:
            return False, "FREE_DAY_LIMIT"
        if user["weekly_used"] >= FREE_WEEKLY_LIMIT:
            return False, "FREE_WEEK_LIMIT"
        return True, None

    weekly_limit = PLAN_WEEKLY_LIMITS.get(plan)
    if weekly_limit is not None and user["weekly_used"] >= weekly_limit:
        return False, f"{plan}_WEEK_LIMIT"

    return True, None


async def increase_usage(telegram_id):
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE users
            SET bonus_requests = CASE
                    WHEN COALESCE(bonus_requests, 0) > 0 THEN bonus_requests - 1
                    ELSE COALESCE(bonus_requests, 0)
                END,
                daily_used = CASE
                    WHEN COALESCE(bonus_requests, 0) > 0 THEN daily_used
                    ELSE daily_used + 1
                END,
                weekly_used = CASE
                    WHEN COALESCE(bonus_requests, 0) > 0 THEN weekly_used
                    ELSE weekly_used + 1
                END
            WHERE telegram_id=$1
        """, telegram_id)


async def check_free_image_limit(user):
    if (user["plan"] or "FREE") != "FREE":
        return True
    return (user["daily_image_used"] or 0) < FREE_DAILY_IMAGE_LIMIT


async def increase_image_usage(telegram_id):
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE users
            SET daily_image_used = daily_image_used + 1
            WHERE telegram_id=$1
        """, telegram_id)


async def save_message(telegram_id, role, content):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO messages (telegram_id, role, content) VALUES ($1, $2, $3)",
            telegram_id,
            role,
            content[:12000],
        )


async def get_chat_history(telegram_id, limit=TEXT_HISTORY_LIMIT):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT role, content
            FROM messages
            WHERE telegram_id=$1
            ORDER BY created_at DESC
            LIMIT $2
        """, telegram_id, limit)

    history = []
    for row in reversed(rows):
        if row["role"] in {"user", "assistant", "system"}:
            history.append({"role": row["role"], "content": row["content"]})
    return history


async def clear_chat(telegram_id):
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM messages WHERE telegram_id=$1", telegram_id)
    await log_event(telegram_id, "delete_context")


async def activate_plan(telegram_id: int, plan: str, months: int):
    until = add_months_rough(months)
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE users
            SET plan=$2, plan_until=$3, daily_used=0, weekly_used=0, daily_image_used=0
            WHERE telegram_id=$1
        """, telegram_id, plan, until)
    await log_event(telegram_id, "activate_plan", f"{plan} {months} months")


async def user_profile_text(user):
    model_names = {
        "gpt": "ChatGPT",
        "claude": "Claude",
        "gemini": "Gemini",
        "nanobanana": "Nano Banana",
        "gptimage": "Sora GPT Image",
        "deepseek": "DeepSeek",
    }

    plan = user["plan"] or "FREE"
    current_model = model_display_name(user["selected_model"])

    if plan == "VIP":
        usage_block = "Запросов: ♾ безлимит"
    elif plan in {"PLUS", "PRO"}:
        usage_block = f"Запросов в неделю: {user['weekly_used']}/{PLAN_WEEKLY_LIMITS[plan]}"
    else:
        usage_block = (
            f"Запросов сегодня: {user['daily_used']}/{FREE_DAILY_LIMIT}\n"
            f"Image сегодня: {user['daily_image_used'] or 0}/{FREE_DAILY_IMAGE_LIMIT}\n"
            f"Запросов в неделю: {user['weekly_used']}/{FREE_WEEKLY_LIMIT}"
        )

    bonus_requests = dict(user).get("bonus_requests") or 0
    if bonus_requests > 0:
        usage_block += f"\nБонусные запросы: {bonus_requests}"

    text = (
        f"📊 Статистика использования\n\n"
        f"{usage_block}\n\n"
        f"Подписка: {plan}\n"
        f"Выбрана модель: {current_model}\n\n"
    )

    if plan != "FREE" and user["plan_until"]:
        text += f"Активна до: {user['plan_until'].strftime('%d.%m.%Y')}\n\n"

    referral_stats = await get_referral_stats(user["telegram_id"])
    text += (
        "💰 Заработать:\n"
        f"• Приглашено: {referral_stats['invited']}\n"
        f"• Купили подписку: {referral_stats['paid']}\n"
        f"• Бонусов получено: {referral_stats['rewards_count']}\n\n"
    )

    text += (
        "Нужно больше? 🚀 Выберите тариф для покупки Premium:\n\n"
        "⭐ PLUS — Claude Sonnet и 500 запросов в неделю\n"
        "💎 PRO — Nano Banana Pro и 1400 запросов в неделю\n"
        "👑 VIP — Sora GPT Image и безлимит"
    )
    return text


async def download_telegram_photo(message: Message) -> bytes:
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    buffer = BytesIO()
    await bot.download_file(file.file_path, destination=buffer)
    return buffer.getvalue()


async def download_telegram_document(message: Message) -> tuple[str, bytes]:
    document = message.document
    filename = document.file_name or "file"

    if document.file_size and document.file_size > MAX_DOCUMENT_SIZE:
        raise ValueError("FILE_TOO_LARGE")

    file = await bot.get_file(document.file_id)
    buffer = BytesIO()
    await bot.download_file(file.file_path, destination=buffer)
    return filename, buffer.getvalue()


async def download_telegram_voice(message: Message) -> tuple[str, bytes]:
    voice = message.voice

    if not voice:
        raise ValueError("VOICE_NOT_FOUND")
    if voice.file_size and voice.file_size > MAX_VOICE_SIZE:
        raise ValueError("VOICE_TOO_LARGE")

    file = await bot.get_file(voice.file_id)
    buffer = BytesIO()
    await bot.download_file(file.file_path, destination=buffer)
    return "voice.ogg", buffer.getvalue()


# File extraction lives in app/ai/file_router.py.

def short_error_text(error: Exception) -> str:
    return str(error).replace("\n", " ")[:1200]


async def send_ai_error_to_admin(error_text: str):
    # По умолчанию НЕ отправляем страшные API-ошибки в Telegram.
    # Они остаются в Railway logs. Если нужно включить диагностику: ADMIN_ERROR_NOTIFICATIONS=true.
    if not ADMIN_ERROR_NOTIFICATIONS:
        return

    friendly_text = (
        "🛠 Диагностика VSotahBot\n\n"
        "Один из AI-провайдеров временно вернул ошибку. Пользователь увидел мягкое сообщение.\n\n"
        f"Технически: {error_text[:1500]}"
    )

    for admin_id in ADMIN_IDS:
        try:
            await bot.send_message(admin_id, friendly_text)
        except Exception:
            pass


async def safe_edit_or_send(callback: CallbackQuery, text: str, reply_markup=None, parse_mode=None):
    try:
        await callback.message.edit_text(
            text,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
            link_preview_options=no_preview(),
        )
    except Exception:
        await callback.message.answer(
            text,
            reply_markup=reply_markup,
            parse_mode=parse_mode,
            link_preview_options=no_preview(),
        )


@dp.message(CommandStart())
async def start_handler(message: Message):
    await clear_start_for_chat(message.chat.id)

    # В приватном чате удаляем пользовательскую команду /start, чтобы она не висела
    # под приветствием. Если Telegram не даст удалить — просто игнорируем.
    if message.chat.type == "private":
        try:
            await message.delete()
        except Exception:
            pass

    now = time.time()
    last = recent_starts.get(message.from_user.id, 0)
    if now - last < 2:
        return
    recent_starts[message.from_user.id] = now

    referrer_id = parse_referral_code(message.text or "")
    await track_referral_start(message.from_user.id, referrer_id)

    user = await get_or_create_user(message)
    await log_event(message.from_user.id, "start")
    await message.answer(
        f"{welcome_text()}\n\nТекущая нейросеть: {model_display_name(user['selected_model'])}",
        reply_markup=main_menu(),
        parse_mode="HTML",
        link_preview_options=no_preview(),
    )


@dp.message(Command("account"))
async def account_command(message: Message):
    user = await get_or_create_user(message)
    await log_event(message.from_user.id, "account")
    await message.answer(await user_profile_text(user), reply_markup=main_menu())


@dp.message(Command("referral"))
async def referral_command(message: Message):
    await get_or_create_user(message)
    await log_event(message.from_user.id, "referral_open")
    bot_username = await get_bot_username(message.bot)
    await message.answer(
        await build_referral_text(message.bot, message.from_user.id),
        reply_markup=referral_menu(bot_username, message.from_user.id),
    )


@dp.message(Command("premium"))
async def premium_command(message: Message):
    await log_event(message.from_user.id, "premium_open")
    await message.answer(premium_text(), reply_markup=tariffs_menu())


@dp.message(Command("models"))
async def models_command(message: Message):
    await log_event(message.from_user.id, "models_command")
    await message.answer(
        "🤖 Выберите нейросеть:\n\n"
        "Все базовые модели сейчас доступны бесплатно.",
        reply_markup=models_menu(),
    )


@dp.message(Command("channels"))
async def channels_command(message: Message):
    await log_event(message.from_user.id, "channels_command")
    await message.answer(channels_text(), reply_markup=channels_menu(), parse_mode="HTML", link_preview_options=no_preview())


@dp.message(Command("deletecontext"))
async def delete_context_command(message: Message):
    await clear_chat(message.from_user.id)
    await message.answer("💬 Контекст очищен. Новый чат начат.", reply_markup=main_menu())


@dp.callback_query(F.data == "back_main")
async def back_main_callback(callback: CallbackQuery):
    await callback.answer()
    await safe_edit_or_send(callback, welcome_text(), reply_markup=main_menu(), parse_mode="HTML")


@dp.callback_query(F.data == "profile")
async def profile_callback(callback: CallbackQuery):
    await callback.answer("Открываю профиль...")
    user = await get_or_create_user_by_data(callback.from_user.id, callback.from_user.username, callback.from_user.first_name)
    await log_event(callback.from_user.id, "profile_click")
    await callback.message.answer(await user_profile_text(user), reply_markup=main_menu())


@dp.callback_query(F.data == "channels")
async def channels_callback(callback: CallbackQuery):
    await callback.answer()
    await log_event(callback.from_user.id, "channels_open")
    await safe_edit_or_send(callback, channels_text(), reply_markup=channels_menu(), parse_mode="HTML")


@dp.callback_query(F.data.in_({"premium", "plans"}))
async def premium_callback(callback: CallbackQuery):
    await callback.answer()
    await log_event(callback.from_user.id, "premium_click")
    await safe_edit_or_send(callback, premium_text(), reply_markup=tariffs_menu())


@dp.callback_query(F.data == "models")
async def models_callback(callback: CallbackQuery):
    await callback.answer()
    await log_event(callback.from_user.id, "models_open")
    await safe_edit_or_send(
        callback,
        "🤖 Выберите нейросеть:\n\n"
        "Все базовые модели сейчас доступны бесплатно.",
        reply_markup=models_menu(),
    )


@dp.callback_query(F.data == "earn")
async def earn_callback(callback: CallbackQuery):
    await callback.answer()
    await get_or_create_user_by_data(callback.from_user.id, callback.from_user.username, callback.from_user.first_name)
    await log_event(callback.from_user.id, "referral_open")
    bot_username = await get_bot_username(callback.bot)
    await safe_edit_or_send(
        callback,
        await build_referral_text(callback.bot, callback.from_user.id),
        reply_markup=referral_menu(bot_username, callback.from_user.id),
    )


@dp.callback_query(F.data == "earn_invite")
async def earn_invite_callback(callback: CallbackQuery):
    await callback.answer()
    await get_or_create_user_by_data(callback.from_user.id, callback.from_user.username, callback.from_user.first_name)
    await log_event(callback.from_user.id, "referral_invite")
    bot_username = await get_bot_username(callback.bot)
    await safe_edit_or_send(
        callback,
        await build_referral_invite_text(callback.bot, callback.from_user.id),
        reply_markup=referral_back_menu(bot_username, callback.from_user.id),
    )


@dp.callback_query(F.data == "earn_top")
async def earn_top_callback(callback: CallbackQuery):
    await callback.answer()
    await log_event(callback.from_user.id, "referral_top")
    bot_username = await get_bot_username(callback.bot)
    await safe_edit_or_send(
        callback,
        await build_referral_leaderboard_text(),
        reply_markup=referral_back_menu(bot_username, callback.from_user.id),
    )


@dp.callback_query(F.data == "earn_bonuses")
async def earn_bonuses_callback(callback: CallbackQuery):
    await callback.answer()
    await get_or_create_user_by_data(callback.from_user.id, callback.from_user.username, callback.from_user.first_name)
    await log_event(callback.from_user.id, "referral_bonuses")
    bot_username = await get_bot_username(callback.bot)
    await safe_edit_or_send(
        callback,
        await build_referral_stats_text(callback.from_user.id),
        reply_markup=referral_back_menu(bot_username, callback.from_user.id),
    )


@dp.callback_query(F.data == "earn_stats")
async def earn_stats_callback(callback: CallbackQuery):
    await callback.answer()
    await get_or_create_user_by_data(callback.from_user.id, callback.from_user.username, callback.from_user.first_name)
    await log_event(callback.from_user.id, "referral_stats")
    bot_username = await get_bot_username(callback.bot)
    await safe_edit_or_send(
        callback,
        await build_referral_stats_text(callback.from_user.id),
        reply_markup=referral_back_menu(bot_username, callback.from_user.id),
    )


@dp.callback_query(F.data.startswith("tariff_"))
async def tariff_callback(callback: CallbackQuery):
    await callback.answer()
    plan = callback.data.replace("tariff_", "")
    if plan not in TARIFFS:
        return
    await log_event(callback.from_user.id, "tariff_select", plan)
    tariff = TARIFFS[plan]
    await safe_edit_or_send(
        callback,
        f"🚀 {tariff['title']}\n\n{tariff['description']}\n\nВыберите период подписки:",
        reply_markup=period_menu(plan),
    )


@dp.callback_query(F.data.startswith("period_"))
async def period_callback(callback: CallbackQuery):
    await callback.answer()
    _, plan, months = callback.data.split("_")
    months = int(months)
    if plan not in TARIFFS:
        return
    price = TARIFFS[plan]["prices"][months]
    await log_event(callback.from_user.id, "period_select", f"{plan} {months}")
    await safe_edit_or_send(
        callback,
        f"💳 Выберите способ оплаты:\n\nТариф: {plan}\nПериод: {months} мес.\nЦена в Stars: ⭐ {price}",
        reply_markup=payment_method_menu(plan, months),
    )


@dp.callback_query(F.data == "rub_disabled")
async def rub_disabled_callback(callback: CallbackQuery):
    await callback.answer("Оплата рублями скоро будет доступна", show_alert=True)


@dp.callback_query(F.data.startswith("pay_stars_"))
async def pay_stars_callback(callback: CallbackQuery):
    await callback.answer()
    _, _, plan, months = callback.data.split("_")
    months = int(months)
    if plan not in TARIFFS:
        return
    price = TARIFFS[plan]["prices"][months]
    payload = f"plan:{plan}:months:{months}:user:{callback.from_user.id}:ts:{int(time.time())}"
    await log_event(callback.from_user.id, "invoice_open", f"{plan} {months} {price}")
    await bot.send_invoice(
        chat_id=callback.message.chat.id,
        title=f"{plan} на {months} мес.",
        description=f"{TARIFFS[plan]['description']}. Подписка на {months} мес.",
        payload=payload,
        provider_token="",
        currency="XTR",
        prices=[LabeledPrice(label=f"{plan} на {months} мес.", amount=price)],
    )


@dp.pre_checkout_query()
async def pre_checkout_handler(pre_checkout_query: PreCheckoutQuery):
    payload = pre_checkout_query.invoice_payload
    if not payload.startswith("plan:"):
        await pre_checkout_query.answer(ok=False, error_message="Некорректный платёж.")
        return
    await pre_checkout_query.answer(ok=True)


@dp.message(F.successful_payment)
async def successful_payment_handler(message: Message):
    payment = message.successful_payment
    payload = payment.invoice_payload
    parts = payload.split(":")
    plan = None
    months = 1
    try:
        plan = parts[1]
        months = int(parts[3])
    except Exception:
        pass

    if plan not in TARIFFS:
        await message.answer("⚠️ Платёж получен, но тариф не распознан. Напишите администратору.")
        return

    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO payments (
                telegram_id, plan, months, amount, currency, payload,
                telegram_payment_charge_id, provider_payment_charge_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            message.from_user.id,
            plan,
            months,
            payment.total_amount,
            payment.currency,
            payload,
            payment.telegram_payment_charge_id,
            payment.provider_payment_charge_id,
        )

    await activate_plan(message.from_user.id, plan, months)
    referrer_id = await mark_referral_paid(message.from_user.id, plan)

    if referrer_id:
        try:
            await bot.send_message(
                referrer_id,
                "🎉 По твоей реферальной ссылке купили подписку!\n\n"
                "Если достигнут новый порог, бонус уже начислен автоматически.",
                reply_markup=main_menu(),
            )
        except Exception:
            pass

    await message.answer(f"✅ Оплата прошла успешно!\n\nТариф {plan} активирован на {months} мес.", reply_markup=main_menu())


@dp.callback_query(F.data.startswith("set_model_"))
async def set_model_callback(callback: CallbackQuery):
    await callback.answer()
    model = callback.data.replace("set_model_", "")
    allowed_models = {"gpt", "claude", "gemini", "nanobanana", "gptimage"}

    if model not in allowed_models:
        await safe_edit_or_send(callback, "⚠️ Эта модель сейчас недоступна.", reply_markup=models_menu())
        return

    user = await get_or_create_user_by_data(callback.from_user.id, callback.from_user.username, callback.from_user.first_name)
    if not has_model_access(user["plan"], model):
        await safe_edit_or_send(callback, premium_required_text(model), reply_markup=tariffs_menu())
        return

    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE users SET selected_model=$1 WHERE telegram_id=$2", model, callback.from_user.id)

    await log_event(callback.from_user.id, "model_select", model)

    try:
        await callback.message.delete()
    except Exception:
        pass

    await bot.send_message(
        chat_id=callback.message.chat.id,
        text=f"✅ Нейросеть выбрана:\n\n{model_display_name(model)}\n\nНапишите запрос, отправьте фото или выберите действие ниже.",
        reply_markup=main_menu(),
    )


@dp.message(Command("admin"))
async def admin_handler(message: Message):
    if not is_admin(message.from_user.id):
        return
    await message.answer(
        "🛠 Админ-панель\n\n"
        "/stats — общая статистика\n"
        "/users — последние пользователи\n"
        "/payments — платежи\n"
        "/refstats — партнёрка\n"
        "/health — статус сервиса\n"
        "/errors — ошибки и лимиты\n"
        "/setplus telegram_id\n"
        "/setpro telegram_id\n"
        "/setvip telegram_id\n"
        "/setfree telegram_id"
    )


def percent(part: int | float | None, total: int | float | None) -> str:
    if not total:
        return "0%"
    return f"{(float(part or 0) / float(total) * 100):.1f}%"


def fmt_dt(value) -> str:
    if not value:
        return "—"
    try:
        return value.strftime("%d.%m %H:%M")
    except Exception:
        return str(value)


@dp.message(Command("stats"))
async def stats_handler(message: Message):
    if not is_admin(message.from_user.id):
        return

    async with db_pool.acquire() as conn:
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        new_users_today = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at::date = CURRENT_DATE")
        active_today = await conn.fetchval("""
            SELECT COUNT(DISTINCT telegram_id)
            FROM events
            WHERE telegram_id IS NOT NULL AND created_at::date = CURRENT_DATE
        """)
        active_24h = await conn.fetchval("""
            SELECT COUNT(DISTINCT telegram_id)
            FROM events
            WHERE telegram_id IS NOT NULL AND created_at >= NOW() - INTERVAL '24 hours'
        """)
        active_7d = await conn.fetchval("""
            SELECT COUNT(DISTINCT telegram_id)
            FROM events
            WHERE telegram_id IS NOT NULL AND created_at >= NOW() - INTERVAL '7 days'
        """)
        starts_today = await conn.fetchval("SELECT COUNT(*) FROM events WHERE event_type='start' AND created_at::date = CURRENT_DATE")
        starts_24h = await conn.fetchval("SELECT COUNT(*) FROM events WHERE event_type='start' AND created_at >= NOW() - INTERVAL '24 hours'")
        total_messages = await conn.fetchval("SELECT COUNT(*) FROM messages WHERE role='user'")
        today_messages = await conn.fetchval("SELECT COUNT(*) FROM messages WHERE role='user' AND created_at::date = CURRENT_DATE")
        messages_24h = await conn.fetchval("SELECT COUNT(*) FROM messages WHERE role='user' AND created_at >= NOW() - INTERVAL '24 hours'")
        vision_requests = await conn.fetchval("SELECT COUNT(*) FROM events WHERE event_type='ai_vision'")
        image_requests = await conn.fetchval("SELECT COUNT(*) FROM events WHERE event_type IN ('ai_image', 'ai_image_edit')")
        file_requests = await conn.fetchval("SELECT COUNT(*) FROM events WHERE event_type='ai_file'")
        voice_requests = await conn.fetchval("SELECT COUNT(*) FROM events WHERE event_type='ai_voice'")
        plus_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE plan='PLUS'")
        pro_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE plan='PRO'")
        vip_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE plan='VIP'")
        premium_clicks = await conn.fetchval("SELECT COUNT(*) FROM events WHERE event_type IN ('premium_click', 'premium_open')")
        invoices = await conn.fetchval("SELECT COUNT(*) FROM events WHERE event_type='invoice_open'")
        payments_count = await conn.fetchval("SELECT COUNT(*) FROM payments")
        total_stars = await conn.fetchval("SELECT COALESCE(SUM(amount), 0) FROM payments")
        payments_today = await conn.fetchval("SELECT COUNT(*) FROM payments WHERE created_at::date = CURRENT_DATE")
        stars_today = await conn.fetchval("SELECT COALESCE(SUM(amount), 0) FROM payments WHERE created_at::date = CURRENT_DATE")
        referral_invites = await conn.fetchval("SELECT COUNT(*) FROM referrals")
        referral_invites_today = await conn.fetchval("SELECT COUNT(*) FROM referrals WHERE created_at::date = CURRENT_DATE")
        referral_paid = await conn.fetchval("SELECT COUNT(*) FROM referrals WHERE status='paid'")
        referral_rewards_count = await conn.fetchval("SELECT COUNT(*) FROM referral_rewards")
        errors_today = await conn.fetchval("""
            SELECT COUNT(*) FROM events
            WHERE created_at::date = CURRENT_DATE
              AND (event_type ILIKE '%error%' OR event_type IN ('ai_provider_error','file_error','vision_error','image_error','voice_error'))
        """)

        model_rows = await conn.fetch("""
            SELECT details AS model, COUNT(*) AS count
            FROM events
            WHERE event_type IN ('ai_message', 'ai_vision', 'ai_image', 'ai_image_edit', 'ai_file', 'ai_voice')
              AND created_at::date = CURRENT_DATE
            GROUP BY details
            ORDER BY count DESC
        """)

    model_text = ""
    if model_rows:
        model_text = "\n".join(f"• {row['model'] or 'unknown'}: {row['count']}" for row in model_rows)
    else:
        model_text = "• сегодня запросов по моделям пока нет"

    await message.answer(
        "📊 Статистика VSotahBot\n\n"
        f"👥 Пользователи всего: {total_users}\n"
        f"🆕 Новых сегодня: {new_users_today}\n"
        f"🔥 Активных сегодня: {active_today}\n"
        f"⏱ Активных за 24 часа: {active_24h}\n"
        f"📅 Активных за 7 дней: {active_7d}\n\n"
        f"🚀 Стартов сегодня: {starts_today}\n"
        f"🚀 Стартов за 24 часа: {starts_24h}\n\n"
        f"💬 Сообщений всего: {total_messages}\n"
        f"💬 Сообщений сегодня: {today_messages}\n"
        f"💬 Сообщений за 24 часа: {messages_24h}\n"
        f"📷 Фото-запросов всего: {vision_requests}\n"
        f"🖼 Image-запросов всего: {image_requests}\n"
        f"📎 Файл-запросов всего: {file_requests}\n"
        f"🎙 Voice-запросов всего: {voice_requests}\n\n"
        f"⭐ PLUS: {plus_users}\n"
        f"💎 PRO: {pro_users}\n"
        f"👑 VIP: {vip_users}\n\n"
        f"💳 Открытий премиума: {premium_clicks}\n"
        f"⭐ Открытий оплаты Stars: {invoices}\n"
        f"✅ Платежей: {payments_count}\n"
        f"⭐ Stars получено: {total_stars}\n\n"
        f"💰 Партнёрка:\n"
        f"• приглашений: {referral_invites}\n"
        f"• оплат от приглашённых: {referral_paid}\n"
        f"• бонусов выдано: {referral_rewards_count}\n\n"
        f"🤖 Модели сегодня:\n{model_text}"
    )


@dp.message(Command("refstats"))
async def refstats_handler(message: Message):
    if not is_admin(message.from_user.id):
        return

    async with db_pool.acquire() as conn:
        total_referrals = await conn.fetchval("SELECT COUNT(*) FROM referrals")
        paid_referrals = await conn.fetchval("SELECT COUNT(*) FROM referrals WHERE status='paid'")
        rewards_count = await conn.fetchval("SELECT COUNT(*) FROM referral_rewards")
        bonus_requests_total = await conn.fetchval("SELECT COALESCE(SUM(bonus_requests), 0) FROM users")
        abuse_events = await conn.fetchval("SELECT COUNT(*) FROM referral_abuse_events")
        recent_reward_rows = await conn.fetch("""
            SELECT referrer_id, milestone, reward_title, source_count, created_at
            FROM referral_rewards
            ORDER BY created_at DESC
            LIMIT 8
        """)
        top_rows = await conn.fetch("""
            SELECT r.referrer_id,
                   COALESCE(u.username, u.first_name, r.referrer_id::text) AS name,
                   COUNT(*) AS invited,
                   COUNT(*) FILTER (WHERE r.status='paid') AS paid
            FROM referrals r
            LEFT JOIN users u ON u.telegram_id = r.referrer_id
            GROUP BY r.referrer_id, u.username, u.first_name
            ORDER BY invited DESC, paid DESC, r.referrer_id
            LIMIT 10
        """)

    def clean_name(value):
        value = str(value or "партнёр")
        if value.isdigit():
            return f"ID {value}"
        if not value.startswith("@") and " " not in value:
            return f"@{value}"
        return value

    top_text = "Пока нет партнёров."
    if top_rows:
        top_text = "\n".join(
            f"{index}. {clean_name(row['name'])} — {row['invited']} приглаш., {row['paid']} оплат"
            for index, row in enumerate(top_rows, start=1)
        )

    recent_rewards_text = "Пока нет выданных бонусов."
    if recent_reward_rows:
        recent_rewards_text = "\n".join(
            f"• ID {row['referrer_id']} — {row['reward_title'] or referral_reward_title(row['milestone'])} ({fmt_dt(row['created_at'])})"
            for row in recent_reward_rows
        )

    await message.answer(
        "💰 Referral stats\n\n"
        f"Всего приглашений: {total_referrals}\n"
        f"Оплат от приглашённых: {paid_referrals}\n"
        f"Конверсия в оплату: {percent(paid_referrals, total_referrals)}\n"
        f"Выдано бонусов: {rewards_count}\n"
        f"Бонусных запросов на балансах: {bonus_requests_total}\n"
        f"Anti-abuse событий: {abuse_events}\n\n"
        f"🏆 Топ партнёров:\n{top_text}\n\n"
        f"🎁 Последние бонусы:\n{recent_rewards_text}"
    )


@dp.message(Command("health"))
async def admin_health_command(message: Message):
    if not is_admin(message.from_user.id):
        return

    started_delta = datetime.utcnow() - STARTED_AT
    uptime_seconds = int(started_delta.total_seconds())
    uptime_text = f"{uptime_seconds // 3600}ч {(uptime_seconds % 3600) // 60}м"

    db_status = "❌ ERROR"
    db_latency_ms = 0
    try:
        start = time.perf_counter()
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_latency_ms = round((time.perf_counter() - start) * 1000)
        db_status = "✅ OK"
    except Exception as e:
        db_status = f"❌ {short_error_text(e)[:120]}"

    bot_status = "❌ ERROR"
    bot_username_text = "—"
    try:
        me = await get_bot_me_cached()
        bot_username_text = f"@{me.username}" if me and me.username else "—"
        bot_status = "✅ OK"
    except Exception as e:
        bot_status = f"❌ {short_error_text(e)[:120]}"

    tavily_status = "❌ OFF"
    if TAVILY_API_KEY:
        try:
            test_results = await search_web("OpenAI latest news")
            tavily_status = "✅ ON / search OK" if test_results else "⚠️ KEY SET / no results"
        except Exception as e:
            tavily_status = f"❌ ERROR: {short_error_text(e)[:90]}"

    await message.answer(
        "🩺 VSotahBot Health\n\n"
        f"Bot API: {bot_status}\n"
        f"Bot: {bot_username_text}\n"
        f"PostgreSQL: {db_status}\n"
        f"DB latency: {db_latency_ms} ms\n"
        f"Uptime: {uptime_text}\n\n"
        f"OpenAI key: {'✅' if OPENAI_API_KEY else '❌'}\n"
        f"Claude key: {'✅' if ANTHROPIC_API_KEY else '❌'}\n"
        f"Gemini key: {'✅' if GOOGLE_API_KEY else '❌'}\n"
        f"DeepSeek key: {'✅' if DEEPSEEK_API_KEY else '❌'}\n"
        f"Live Web / Tavily: {tavily_status}\n"
        f"Admin error notifications: {'ON' if ADMIN_ERROR_NOTIFICATIONS else 'OFF'}"
    )


@dp.message(Command("errors"))
async def admin_errors_command(message: Message):
    if not is_admin(message.from_user.id):
        return

    async with db_pool.acquire() as conn:
        error_rows = await conn.fetch("""
            SELECT event_type, details, telegram_id, created_at
            FROM events
            WHERE event_type ILIKE '%error%'
               OR event_type IN ('ai_provider_error','file_error','vision_error','image_error','voice_error')
            ORDER BY created_at DESC
            LIMIT 15
        """)
        limit_rows = await conn.fetch("""
            SELECT details, COUNT(*) AS count
            FROM events
            WHERE event_type='limit_reached' AND created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY details
            ORDER BY count DESC
            LIMIT 8
        """)

    errors_text = "Ошибок пока нет."
    if error_rows:
        errors_text = "\n".join(
            f"• {fmt_dt(row['created_at'])} | {row['event_type']} | ID {row['telegram_id'] or '—'} | {(row['details'] or '')[:140]}"
            for row in error_rows
        )

    limits_text = "Лимиты за 24 часа не упирались."
    if limit_rows:
        limits_text = "\n".join(
            f"• {row['details'] or 'unknown'}: {row['count']}"
            for row in limit_rows
        )

    await message.answer(
        "⚠️ VSotah Errors / Limits\n\n"
        f"Последние ошибки:\n{errors_text}\n\n"
        f"Лимиты за 24 часа:\n{limits_text}"
    )


@dp.message(Command("users"))
async def users_handler(message: Message):
    if not is_admin(message.from_user.id):
        return

    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT telegram_id, username, first_name, plan, weekly_used, created_at
            FROM users ORDER BY created_at DESC LIMIT 15
        """)

    text = "👥 Последние пользователи\n\n"
    for row in rows:
        name = row["username"] or row["first_name"] or "без имени"
        text += (
            f"ID: {row['telegram_id']}\n"
            f"Имя: {name}\n"
            f"Тариф: {row['plan']}\n"
            f"Запросов за неделю: {row['weekly_used']}\n"
            f"Дата: {row['created_at'].strftime('%d.%m %H:%M')}\n\n"
        )
    await message.answer(text[:3900])


@dp.message(Command("payments"))
async def payments_handler(message: Message):
    if not is_admin(message.from_user.id):
        return

    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT telegram_id, plan, months, amount, currency, created_at
            FROM payments ORDER BY created_at DESC LIMIT 15
        """)

    if not rows:
        await message.answer("Платежей пока нет.")
        return

    text = "💳 Последние платежи Telegram Stars\n\n"
    for row in rows:
        text += (
            f"ID: {row['telegram_id']}\n"
            f"Тариф: {row['plan']} на {row['months']} мес.\n"
            f"Сумма: {row['amount']} {row['currency']}\n"
            f"Дата: {row['created_at'].strftime('%d.%m %H:%M')}\n\n"
        )
    await message.answer(text[:3900])


async def admin_set_plan(message: Message, plan: str):
    if not is_admin(message.from_user.id):
        return

    parts = message.text.split()
    if len(parts) != 2 or not parts[1].isdigit():
        await message.answer("Формат команды: /setpro telegram_id")
        return

    telegram_id = int(parts[1])
    until = add_months_rough(1) if plan != "FREE" else None

    async with db_pool.acquire() as conn:
        await conn.execute("UPDATE users SET plan=$2, plan_until=$3 WHERE telegram_id=$1", telegram_id, plan, until)

    await log_event(telegram_id, "admin_set_plan", plan)
    await message.answer(f"✅ Пользователю {telegram_id} установлен тариф {plan}.")


@dp.message(Command("setplus"))
async def setplus_handler(message: Message):
    await admin_set_plan(message, "PLUS")


@dp.message(Command("setpro"))
async def setpro_handler(message: Message):
    await admin_set_plan(message, "PRO")


@dp.message(Command("setvip"))
async def setvip_handler(message: Message):
    await admin_set_plan(message, "VIP")


@dp.message(Command("setfree"))
async def setfree_handler(message: Message):
    await admin_set_plan(message, "FREE")



async def run_work_ai_command(message: Message, mode: str, router_func, title: str, wait_text: str):
    """AI Core Upgrade: serious work modes without adding heavy UI."""
    raw_text = message.text or ""
    prompt = raw_text.split(maxsplit=1)[1].strip() if len(raw_text.split(maxsplit=1)) > 1 else ""

    if not prompt:
        examples = {
            "research": "Например: /research рынок Telegram AI-ботов в 2026 и как продвигать VSotahBot",
            "business": "Например: /business напиши коммерческое предложение для рекламы VSotahBot в каналах",
            "code": "Например: /code объясни ошибку Railway и предложи готовое исправление",
            "web": "Например: /web последние новости OpenAI и Gemini сегодня",
        }
        await message.answer(
            f"{title}\n\nНапишите запрос после команды.\n\n{examples.get(mode, '')}",
            reply_markup=main_menu(),
        )
        return

    user = await get_or_create_user(message)
    spam_allowed, wait_seconds = await check_spam(message.from_user.id)
    if not spam_allowed:
        await message.answer(f"🛡 Слишком много сообщений подряд.\n\nПопробуйте снова через {wait_seconds} сек.")
        return

    allowed, reason = await check_limit(user)
    if not allowed:
        await log_event(message.from_user.id, "limit_reached", reason)
        await message.answer("⏳ Лимит сообщений закончился.\n\nВы можете перейти на PLUS, PRO или VIP.", reply_markup=tariffs_menu())
        return

    selected_model = user["selected_model"]
    if selected_model in {"nanobanana", "gptimage"}:
        selected_model = "gpt"

    if not has_model_access(user["plan"], selected_model):
        selected_model = "gpt"
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE users SET selected_model='gpt' WHERE telegram_id=$1", message.from_user.id)

    wait_message = await message.answer(wait_text)

    try:
        await save_message(message.from_user.id, "user", f"[{mode.upper()}] {prompt}")
        history = await get_chat_history(message.from_user.id)

        try:
            await wait_message.edit_text("⚡ Собираю контекст и думаю...")
        except Exception:
            pass

        answer = await router_func(selected_model, history)
        if not answer:
            answer = "⚠️ AI вернул пустой ответ. Попробуйте переформулировать запрос."

        await save_message(message.from_user.id, "assistant", answer)
        await increase_usage(message.from_user.id)
        await log_event(message.from_user.id, f"ai_{mode}", selected_model)

        if len(answer) <= 3900:
            await wait_message.edit_text(answer)
        else:
            await wait_message.edit_text(answer[:3900])
            for i in range(3900, len(answer), 3900):
                await message.answer(answer[i:i + 3900])

    except Exception as e:
        admin_error = short_error_text(e)
        print(f"{mode.upper()} AI ERROR SHORT:\n{admin_error}")
        print(f"{mode.upper()} AI ERROR TRACE:\n{traceback.format_exc()}")
        await log_event(message.from_user.id, f"{mode}_error", admin_error)
        await send_ai_error_to_admin(f"⚠️ {mode.upper()} AI ERROR | {selected_model} | {admin_error}")
        try:
            await wait_message.edit_text(
                "⚠️ Не удалось обработать запрос. Попробуйте позже или выберите другую нейросеть.",
                reply_markup=main_menu(),
            )
        except Exception:
            await message.answer("⚠️ Не удалось обработать запрос.", reply_markup=main_menu())


@dp.message(Command("research"))
async def research_command(message: Message):
    await run_work_ai_command(
        message,
        "research",
        research_router,
        "🔎 Deep Research Lite",
        "🔎 Ищу и анализирую информацию...",
    )


@dp.message(Command("web"))
async def web_command(message: Message):
    await run_work_ai_command(
        message,
        "web",
        web_router,
        "🌐 Live Web AI",
        "🌐 Проверяю актуальную информацию...",
    )


@dp.message(Command("business"))
async def business_command(message: Message):
    await run_work_ai_command(
        message,
        "business",
        business_router,
        "💼 Business AI",
        "💼 Готовлю рабочий ответ...",
    )


@dp.message(Command("code"))
async def code_command(message: Message):
    await run_work_ai_command(
        message,
        "code",
        code_router,
        "💻 Code AI",
        "💻 Анализирую код/ошибку...",
    )


@dp.message(F.photo)
async def photo_handler(message: Message):
    user = await get_or_create_user(message)
    spam_allowed, wait_seconds = await check_spam(message.from_user.id)

    if not spam_allowed:
        await message.answer(f"🛡 Слишком много сообщений подряд.\n\nПопробуйте снова через {wait_seconds} сек.")
        return

    allowed, reason = await check_limit(user)
    if not allowed:
        await log_event(message.from_user.id, "limit_reached", reason)
        await message.answer("⏳ Лимит сообщений закончился.\n\nВы можете перейти на PLUS, PRO или VIP.", reply_markup=tariffs_menu())
        return

    selected_model = user["selected_model"]
    if not has_model_access(user["plan"], selected_model):
        selected_model = "gpt"
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE users SET selected_model='gpt' WHERE telegram_id=$1", message.from_user.id)
        await message.answer(
            "ℹ️ Ваш тариф изменился, поэтому я переключил нейросеть на ChatGPT — GPT-4o mini."
        )

    is_image_edit = selected_model == "gptimage"
    if is_image_edit and not await check_free_image_limit(user):
        await log_event(message.from_user.id, "limit_reached", "FREE_IMAGE_DAY_LIMIT")
        await message.answer(
            "🖼 Вы исчерпали бесплатный лимит генерации изображений.\n\n"
            "FREE: 5 Image в день. Перейдите на PLUS, PRO или VIP, чтобы получить больше лимитов.",
            reply_markup=tariffs_menu(),
        )
        return

    wait_message = await message.answer("🖼 Редактирую изображение..." if is_image_edit else "📷 Анализирую фото...")

    try:
        question = message.caption or ("Улучши это изображение и сохрани смысл." if is_image_edit else "Что изображено на фото?")
        image_bytes = await download_telegram_photo(message)

        if is_image_edit:
            await save_message(message.from_user.id, "user", f"[Редактирование изображения] {question}")
            edited_bytes, text_note = await edit_gpt_image(question, image_bytes)

            if not edited_bytes:
                print(f"IMAGE EDIT RETURNED NO IMAGE | {(text_note or '')[:1000]}")
                await wait_message.edit_text(
                    "⚠️ Генерация временно недоступна. Попробуйте ещё раз чуть позже.",
                    reply_markup=main_menu(),
                )
                return

            photo = BufferedInputFile(edited_bytes, filename="edited_gpt_image.png")
            await wait_message.delete()
            await message.answer_photo(photo=photo, caption=(text_note[:900] if text_note else "✅ Готово"))
            await save_message(message.from_user.id, "assistant", "[gptimage image edited]")
            await increase_usage(message.from_user.id)
            await increase_image_usage(message.from_user.id)
            await log_event(message.from_user.id, "ai_image_edit", selected_model)
            return

        await save_message(message.from_user.id, "user", f"[Фото] {question}")
        history = await get_chat_history(message.from_user.id)
        answer = await vision_router(selected_model, question, image_bytes, history)

        if not answer:
            answer = "⚠️ AI не смог проанализировать фото. Попробуйте отправить другое изображение или добавьте вопрос текстом."

        await save_message(message.from_user.id, "assistant", answer)
        await increase_usage(message.from_user.id)
        await log_event(message.from_user.id, "ai_vision", selected_model)

        try:
            await wait_message.edit_text("✅ Готово")
        except Exception:
            pass

        if len(answer) <= 3900:
            await wait_message.edit_text(answer)
        else:
            await wait_message.edit_text(answer[:3900])
            for i in range(3900, len(answer), 3900):
                await message.answer(answer[i:i + 3900])

    except Exception as e:
        admin_error = short_error_text(e)
        print(f"VISION ERROR SHORT:\n{admin_error}")
        print(f"VISION ERROR TRACE:\n{traceback.format_exc()}")
        await log_event(message.from_user.id, "vision_error", admin_error)

        try:
            await wait_message.edit_text(
                "⚠️ Не удалось обработать фото.\n\n"
                "Попробуйте ещё раз или выберите другую нейросеть.",
                reply_markup=main_menu(),
            )
        except Exception:
            await message.answer(
                "⚠️ Не удалось обработать фото.\n\n"
                "Попробуйте ещё раз или выберите другую нейросеть.",
                reply_markup=main_menu(),
            )


@dp.message(F.document)
async def document_handler(message: Message):
    user = await get_or_create_user(message)
    spam_allowed, wait_seconds = await check_spam(message.from_user.id)

    if not spam_allowed:
        await message.answer(f"🛡 Слишком много сообщений подряд.\n\nПопробуйте снова через {wait_seconds} сек.")
        return

    allowed, reason = await check_limit(user)
    if not allowed:
        await log_event(message.from_user.id, "limit_reached", reason)
        await message.answer("⏳ Лимит сообщений закончился.\n\nВы можете перейти на PLUS, PRO или VIP.", reply_markup=tariffs_menu())
        return

    selected_model = user["selected_model"]
    if selected_model in {"nanobanana", "gptimage"}:
        await message.answer(
            "📎 Для анализа файлов выберите ChatGPT, Claude или Gemini в меню «Выбрать AI».",
            reply_markup=main_menu(),
        )
        return

    if not has_model_access(user["plan"], selected_model):
        selected_model = "gpt"
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE users SET selected_model='gpt' WHERE telegram_id=$1", message.from_user.id)
        await message.answer(
            "ℹ️ Ваш тариф изменился, поэтому я переключил нейросеть на ChatGPT — GPT-4o mini."
        )

    wait_message = await message.answer("📎 Читаю файл...")

    try:
        filename, file_bytes = await download_telegram_document(message)
        question = message.caption or "Проанализируй файл и выдели главное."
        await wait_message.edit_text(build_file_status_text(filename, "reading"))

        try:
            extracted_text = await asyncio.to_thread(extract_document_text, filename, file_bytes)
        except ValueError as e:
            error_code = str(e)
            if error_code == "FILE_TOO_LARGE":
                await wait_message.edit_text("⚠️ Файл слишком большой. Сейчас лимит — до 10 МБ.", reply_markup=main_menu())
            elif error_code == "UNSUPPORTED_FILE_TYPE":
                await wait_message.edit_text("⚠️ Пока поддерживаются TXT, PDF, DOCX и XLSX.", reply_markup=main_menu())
            else:
                await wait_message.edit_text(
                    "⚠️ Не удалось прочитать файл. Попробуйте другой файл или отправьте текстом.",
                    reply_markup=main_menu(),
                )
            return

        await save_message(message.from_user.id, "user", f"[Файл {filename}] {question}")
        history = await get_chat_history(message.from_user.id)

        if not extracted_text.strip():
            if filename.lower().endswith(".pdf"):
                await wait_message.edit_text("📄 PDF похож на скан. Анализирую страницы как изображения...")
                answer = await analyze_pdf_images_with_openai(question, filename, file_bytes, history)
                if not answer:
                    await wait_message.edit_text(
                        "⚠️ Не удалось извлечь текст из PDF. Возможно, файл защищён или страницы плохо читаются.",
                        reply_markup=main_menu(),
                    )
                    return
            else:
                await wait_message.edit_text(
                    "⚠️ Не удалось извлечь текст из файла. Возможно, это скан или защищённый документ.",
                    reply_markup=main_menu(),
                )
                return
        else:
            await wait_message.edit_text(build_file_status_text(filename, "analyzing"))
            answer = await file_router(selected_model, question, filename, extracted_text, history)

        if not answer:
            answer = "⚠️ AI вернул пустой ответ. Попробуйте задать вопрос по файлу точнее."

        await save_message(message.from_user.id, "assistant", answer)
        await increase_usage(message.from_user.id)
        await log_event(message.from_user.id, "ai_file", selected_model)

        try:
            await wait_message.edit_text("✅ Готово")
        except Exception:
            pass

        if len(answer) <= 3900:
            await wait_message.edit_text(answer)
        else:
            await wait_message.edit_text(answer[:3900])
            for i in range(3900, len(answer), 3900):
                await message.answer(answer[i:i + 3900])

    except Exception as e:
        admin_error = short_error_text(e)
        print(f"FILE ERROR SHORT:\n{admin_error}")
        print(f"FILE ERROR TRACE:\n{traceback.format_exc()}")
        await log_event(message.from_user.id, "file_error", admin_error)

        try:
            await wait_message.edit_text(
                "⚠️ Не удалось обработать файл. Попробуйте ещё раз или отправьте файл в другом формате.",
                reply_markup=main_menu(),
            )
        except Exception:
            await message.answer(
                "⚠️ Не удалось обработать файл. Попробуйте ещё раз или отправьте файл в другом формате.",
                reply_markup=main_menu(),
            )


async def send_fast_voice_reply(message: Message, answer: str):
    """Generate a short voice reply in background so text answer stays instant."""
    status_message = None
    try:
        status_message = await message.answer("🔊 Голосовой ответ готовится в фоне...")
        audio_reply = await text_to_speech(answer)
        if not audio_reply:
            if status_message:
                await status_message.edit_text("⚠️ Не удалось подготовить голосовой ответ.")
            return

        filename = "vsotah_voice_reply.ogg"
        audio = BufferedInputFile(audio_reply, filename=filename)
        await message.answer_voice(voice=audio, caption="🔊 Голосовой ответ")

        if status_message:
            try:
                await status_message.delete()
            except Exception:
                await status_message.edit_text("✅ Голосовой ответ отправлен.")

    except Exception as voice_reply_error:
        await log_event(message.from_user.id, "voice_tts_error", short_error_text(voice_reply_error))
        print(f"VOICE TTS ERROR: {short_error_text(voice_reply_error)}")
        if status_message:
            try:
                await status_message.edit_text("⚠️ Голосовой ответ не получился, но текстовый ответ выше уже готов.")
            except Exception:
                pass


@dp.message(F.voice)
async def voice_handler(message: Message):
    """Voice AI 2.0: free voice assistant for all users.

    Voice messages do not consume daily/weekly request limits.
    Text answer is always returned; voice reply is enabled by default and can be disabled
    with VOICE_REPLY_ENABLED=false in Railway Variables.
    """
    user = await get_or_create_user(message)
    spam_allowed, wait_seconds = await check_spam(message.from_user.id)

    if not spam_allowed:
        await message.answer(f"🛡 Слишком много сообщений подряд.\n\nПопробуйте снова через {wait_seconds} сек.")
        return

    selected_model = user["selected_model"]
    if selected_model in {"nanobanana", "gptimage"}:
        selected_model = "gpt"

    if not has_model_access(user["plan"], selected_model):
        selected_model = "gpt"
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE users SET selected_model='gpt' WHERE telegram_id=$1", message.from_user.id)
        await message.answer("ℹ️ Для голосового AI я переключил модель на ChatGPT — GPT-4o mini.")

    wait_message = await message.answer("🎙 Слушаю голосовое...")

    try:
        filename, audio_bytes = await download_telegram_voice(message)

        await wait_message.edit_text("🎧 Распознаю речь...")
        transcript = await transcribe_voice(audio_bytes, filename)

        if not transcript:
            await wait_message.edit_text(
                "⚠️ Не удалось распознать голосовое. Попробуйте записать ещё раз или отправьте текстом.",
                reply_markup=main_menu(),
            )
            return

        await save_message(message.from_user.id, "user", build_voice_user_message(transcript))
        history = await get_chat_history(message.from_user.id)

        await wait_message.edit_text("✍️ Печатает ответ...")
        answer = await ai_router(selected_model, history)

        if not answer:
            answer = "⚠️ AI вернул пустой ответ. Попробуйте записать вопрос ещё раз."

        await save_message(message.from_user.id, "assistant", answer)
        await log_event(message.from_user.id, "ai_voice_free", selected_model)

        # Пользователю не нужен технический блок "Распознал / Ответ".
        # Показываем сразу готовый ответ, как в обычном чате.
        response_text = answer
        if len(response_text) <= 3900:
            await wait_message.edit_text(response_text)
        else:
            await wait_message.edit_text(response_text[:3900])
            for i in range(3900, len(response_text), 3900):
                await message.answer(response_text[i:i + 3900])

        if VOICE_REPLY_ENABLED:
            # Do not block the voice handler while TTS is being generated.
            # The user already has the full text answer; the short audio answer arrives separately.
            asyncio.create_task(send_fast_voice_reply(message, answer))

    except ValueError as e:
        error_code = str(e)
        if error_code == "VOICE_TOO_LARGE":
            await wait_message.edit_text("⚠️ Голосовое слишком большое. Сейчас лимит — до 20 МБ.", reply_markup=main_menu())
        else:
            await wait_message.edit_text("⚠️ Не удалось скачать голосовое. Попробуйте ещё раз.", reply_markup=main_menu())

    except Exception as e:
        admin_error = short_error_text(e)
        print(f"VOICE ERROR SHORT:\n{admin_error}")
        print(f"VOICE ERROR TRACE:\n{traceback.format_exc()}")
        await log_event(message.from_user.id, "voice_error", admin_error)
        await send_ai_error_to_admin(f"⚠️ VOICE AI ERROR | {selected_model} | {admin_error}")

        try:
            await wait_message.edit_text(
                "⚠️ Не удалось обработать голосовое. Попробуйте ещё раз или отправьте текстом.",
                reply_markup=main_menu(),
            )
        except Exception:
            await message.answer(
                "⚠️ Не удалось обработать голосовое. Попробуйте ещё раз или отправьте текстом.",
                reply_markup=main_menu(),
            )


@dp.message(F.text)
async def chat_handler(message: Message):
    if message.text.startswith("/"):
        return

    user = await get_or_create_user(message)
    spam_allowed, wait_seconds = await check_spam(message.from_user.id)

    if not spam_allowed:
        await message.answer(f"🛡 Слишком много сообщений подряд.\n\nПопробуйте снова через {wait_seconds} сек.")
        return

    allowed, reason = await check_limit(user)
    if not allowed:
        await log_event(message.from_user.id, "limit_reached", reason)
        await message.answer("⏳ Лимит сообщений закончился.\n\nВы можете перейти на PLUS, PRO или VIP.", reply_markup=tariffs_menu())
        return

    selected_model = user["selected_model"]
    if not has_model_access(user["plan"], selected_model):
        selected_model = "gpt"
        async with db_pool.acquire() as conn:
            await conn.execute("UPDATE users SET selected_model='gpt' WHERE telegram_id=$1", message.from_user.id)
        await message.answer(
            "ℹ️ Ваш тариф изменился, поэтому я переключил нейросеть на ChatGPT — GPT-4o mini."
        )

    if selected_model in {"nanobanana", "gptimage"}:
        if not await check_free_image_limit(user):
            await log_event(message.from_user.id, "limit_reached", "FREE_IMAGE_DAY_LIMIT")
            await message.answer(
                "🖼 Вы исчерпали бесплатный лимит генерации изображений.\n\n"
                "FREE: 5 Image в день. Перейдите на PLUS, PRO или VIP, чтобы получить больше лимитов.",
                reply_markup=tariffs_menu(),
            )
            return

        wait_text = "🍌 Генерирую изображение..." if selected_model == "nanobanana" else "🌀 Генерирую изображение..."
        wait_message = await message.answer(wait_text)
        try:
            await save_message(message.from_user.id, "user", message.text)

            if selected_model == "nanobanana":
                image_bytes, text_note = await generate_nano_banana_image(message.text)
            else:
                image_bytes, text_note = await generate_gpt_image(message.text)

            if not image_bytes:
                print(f"IMAGE PROVIDER RETURNED NO IMAGE | {selected_model} | {(text_note or '')[:1000]}")
                await wait_message.edit_text(
                    "⚠️ Генерация временно недоступна. Попробуйте ещё раз чуть позже.",
                    reply_markup=main_menu(),
                )
                return

            filename = "nano_banana.png" if selected_model == "nanobanana" else "sora_gpt_image.png"
            photo = BufferedInputFile(image_bytes, filename=filename)
            await wait_message.delete()
            await message.answer_photo(
                photo=photo,
                caption=(text_note[:900] if text_note else "✅ Готово"),
            )

            await save_message(message.from_user.id, "assistant", f"[{selected_model} image generated]")
            await increase_usage(message.from_user.id)
            await increase_image_usage(message.from_user.id)
            await log_event(message.from_user.id, "ai_image", selected_model)
            return

        except Exception as e:
            admin_error = short_error_text(e)
            print(f"IMAGE ERROR SHORT:\n{admin_error}")
            print(f"IMAGE ERROR TRACE:\n{traceback.format_exc()}")
            await log_event(message.from_user.id, "image_error", admin_error)
            try:
                await wait_message.edit_text(
                    "⚠️ Генерация временно недоступна. Попробуйте ещё раз чуть позже.",
                    reply_markup=main_menu(),
                )
            except Exception:
                await message.answer(
                    "⚠️ Генерация временно недоступна. Попробуйте ещё раз чуть позже.",
                    reply_markup=main_menu(),
                )
            return

    # Не отправляем Telegram chat_action typing для обычного текста:
    # на некоторых клиентах он может висеть сверху несколько секунд уже после ответа.
    # Вместо этого используем собственное loading-сообщение, которое точно заменяется итоговым ответом.
    wait_message = await message.answer("✍️ Печатает ответ...")

    try:
        await save_message(message.from_user.id, "user", message.text)
        history = await get_chat_history(message.from_user.id)

        try:
            await wait_message.edit_text("✍️ Печатает ответ...")
        except Exception:
            pass

        answer = await ai_router(selected_model, history)

        if not answer:
            answer = "⚠️ AI вернул пустой ответ. Попробуйте переформулировать вопрос."

        await save_message(message.from_user.id, "assistant", answer)
        await increase_usage(message.from_user.id)
        await log_event(message.from_user.id, "ai_message", selected_model)

        try:
            await wait_message.edit_text("✅ Готово")
        except Exception:
            pass

        if len(answer) <= 3900:
            await wait_message.edit_text(answer)
        else:
            await wait_message.edit_text(answer[:3900])
            for i in range(3900, len(answer), 3900):
                await message.answer(answer[i:i + 3900])

    except Exception as e:
        admin_error = short_error_text(e)
        print(f"AI ERROR SHORT:\n{admin_error}")
        print(f"AI ERROR TRACE:\n{traceback.format_exc()}")
        await log_event(message.from_user.id, "ai_provider_error", admin_error)

        await send_ai_error_to_admin(f"⚠️ AI ERROR | {selected_model} | {admin_error}")

        try:
            await wait_message.edit_text(
                "⚠️ Сейчас выбранная нейросеть временно недоступна.\n\n"
                "Попробуйте позже или выберите другую нейросеть.",
                reply_markup=main_menu(),
            )
        except Exception:
            await message.answer(
                "⚠️ Сейчас выбранная нейросеть временно недоступна.\n\n"
                "Попробуйте позже или выберите другую нейросеть.",
                reply_markup=main_menu(),
            )


async def health_handler(request):
    return web.json_response({"ok": True, "service": "vsotah-bot"})


async def tribute_webhook_handler(request):
    try:
        data = await request.json()
        print(f"TRIBUTE WEBHOOK RECEIVED: {str(data)[:1000]}")
        await log_event(None, "tribute_webhook_received", str(data)[:1000])
    except Exception as e:
        print(f"TRIBUTE WEBHOOK ERROR: {e}")
    return web.json_response({"ok": True})


async def start_web_server():
    app = web.Application()
    app.router.add_get("/", health_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_post("/tribute-webhook", tribute_webhook_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    print(f"WEB SERVER STARTED ON PORT {PORT}")


async def main():
    print("BOT STARTING")

    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is missing")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing")

    await bot.delete_webhook(drop_pending_updates=True)
    await init_db()
    await setup_bot_info()
    await start_web_server()

    print("DATABASE CONNECTED")
    print("BOT STARTED")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())







