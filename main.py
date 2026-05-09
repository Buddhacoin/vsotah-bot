import asyncio
import base64
import json
import os
import secrets
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
    LinkPreviewOptions,
    BufferedInputFile,
)
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types


BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TRIBUTE_API_KEY = os.getenv("TRIBUTE_API_KEY")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://gptclaude-bot-production.up.railway.app").rstrip("/")
PORT = int(os.getenv("PORT", "8080"))

OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
ANTHROPIC_TEXT_MODEL = os.getenv("ANTHROPIC_TEXT_MODEL", "claude-sonnet-4-6")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
NANO_BANANA_MODEL = os.getenv("NANO_BANANA_MODEL", "imagen-4.0-generate-001")
GPT_IMAGE_MODEL = os.getenv("GPT_IMAGE_MODEL", "gpt-image-1")

ADMIN_IDS = {
    int(x.strip())
    for x in os.getenv("ADMIN_IDS", "").split(",")
    if x.strip().isdigit()
}

FREE_DAILY_LIMIT = 15
FREE_WEEKLY_LIMIT = 105

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

TRIBUTE_LINKS = {
    "PLUS": {
        1: "https://web.tribute.tg/p/vJ9",
        3: "https://web.tribute.tg/p/vJc",
        6: "https://web.tribute.tg/p/vJd",
        12: "https://web.tribute.tg/p/vJe",
    },
    "PRO": {
        1: "https://web.tribute.tg/p/vJg",
        3: "https://web.tribute.tg/p/vJh",
        6: "https://web.tribute.tg/p/vJi",
        12: "https://web.tribute.tg/p/vJj",
    },
    "VIP": {
        1: "https://web.tribute.tg/p/vJk",
        3: "https://web.tribute.tg/p/vJl",
        6: "https://web.tribute.tg/p/vJm",
        12: "https://web.tribute.tg/p/vJo",
    },
}

SPAM_WINDOW_SECONDS = 8
SPAM_MAX_MESSAGES = 5
SPAM_BLOCK_SECONDS = 60

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
google_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
) if DEEPSEEK_API_KEY else None

db_pool = None
recent_starts = {}


def no_preview():
    return LinkPreviewOptions(is_disabled=True)


def main_menu():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👤 Профиль", callback_data="profile"),
                InlineKeyboardButton(text="💳 Купить подписку", callback_data="premium"),
            ],
            [InlineKeyboardButton(text="🤖 Выбрать нейросеть", callback_data="models")],
            [InlineKeyboardButton(text="🧠 Наши каналы", callback_data="channels")],
        ]
    )


def models_menu():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🌀 ChatGPT", callback_data="set_model_gpt")],
            [InlineKeyboardButton(text="✴️ Claude", callback_data="set_model_claude")],
            [InlineKeyboardButton(text="✦ Gemini", callback_data="set_model_gemini")],
            [InlineKeyboardButton(text="🍌 Nano Banana", callback_data="set_model_nanobanana")],
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
            [InlineKeyboardButton(text="⚡ Молния News", url="https://t.me/LightningNewsSupport")],
            [InlineKeyboardButton(text="← Назад", callback_data="back_main")],
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
            [InlineKeyboardButton(text="💳 Карта / СБП — скоро", callback_data="rub_payment_disabled")],
            [InlineKeyboardButton(text="⭐ Telegram Stars", callback_data=f"pay_stars_{plan}_{months}")],
            [InlineKeyboardButton(text="← Назад", callback_data=f"tariff_{plan}")],
        ]
    )


def tribute_open_payment_menu(payment_url: str, plan: str, months: int):
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="💳 Открыть оплату картой / СБП", url=payment_url)],
            [InlineKeyboardButton(text="⭐ Оплатить Telegram Stars", callback_data=f"pay_stars_{plan}_{months}")],
            [InlineKeyboardButton(text="← Назад", callback_data=f"period_{plan}_{months}")],
        ]
    )


def get_week_start():
    today = date.today()
    return today - timedelta(days=today.weekday())


def add_months_rough(months: int):
    return datetime.utcnow() + timedelta(days=30 * months)


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


def model_display_name(model: str) -> str:
    names = {
        "gpt": "🌀 ChatGPT",
        "claude": "✴️ Claude",
        "gemini": "✦ Gemini",
        "nanobanana": "🍌 Nano Banana",
        "gptimage": "🌀 Sora GPT Image",
    }
    return names.get(model, "🌀 ChatGPT")


def welcome_text():
    return """👋 Добро пожаловать в @GPTclaudeAIbot

Ваш AI-бот для работы с нейросетями в одном месте.

📝 Генерация текста:
• ChatGPT
• Claude
• Gemini

🌇 Генерация изображений:
• Nano Banana Pro
• Sora GPT Image

🧠 Наши каналы:
• Наш канал: <a href='https://t.me/MolniyaLiveNews'>Молния Live</a>
• Канал support: <a href='https://t.me/LightningNewsSupport'>Молния News</a>

Напишите вопрос или выберите действие ниже."""


def premium_text():
    return """💳 Купить подписку

⭐ PLUS — 500 запросов в неделю
💎 PRO — 1400 запросов в неделю
👑 VIP — безлимит"""


def channels_text():
    return """🧠 Наши каналы:

• Наш канал: <a href='https://t.me/MolniyaLiveNews'>Молния Live</a>
• Канал support: <a href='https://t.me/LightningNewsSupport'>Молния News</a>"""


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
                day_start DATE DEFAULT CURRENT_DATE,
                week_start DATE DEFAULT CURRENT_DATE,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS plan_until TIMESTAMP;")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS selected_model TEXT DEFAULT 'gpt';")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS daily_used INTEGER DEFAULT 0;")
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS weekly_used INTEGER DEFAULT 0;")
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
            CREATE TABLE IF NOT EXISTS tribute_sessions (
                token TEXT PRIMARY KEY,
                telegram_id BIGINT NOT NULL,
                plan TEXT NOT NULL,
                months INTEGER NOT NULL,
                tribute_url TEXT NOT NULL,
                status TEXT DEFAULT 'created',
                created_at TIMESTAMP DEFAULT NOW(),
                clicked_at TIMESTAMP
            );
        """)

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


async def setup_bot_info():
    await bot.set_my_commands([
        BotCommand(command="start", description="👋 Что умеет бот"),
        BotCommand(command="account", description="👤 Мой профиль"),
        BotCommand(command="premium", description="💳 Купить подписку"),
        BotCommand(command="models", description="🤖 Выбрать нейросеть"),
        BotCommand(command="channels", description="🧠 Наши каналы"),
        BotCommand(command="deletecontext", description="💬 Удалить контекст"),
    ])
    # Описание и вступление бота НЕ трогаем из кода.
    # Их настраиваем вручную через BotFather, чтобы деплой не перезаписывал текст.


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
            await conn.execute("UPDATE users SET daily_used=0, day_start=$2 WHERE telegram_id=$1", telegram_id, today)

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

    if plan == "VIP":
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
            SET daily_used = daily_used + 1,
                weekly_used = weekly_used + 1
            WHERE telegram_id=$1
        """, telegram_id)


async def save_message(telegram_id, role, content):
    async with db_pool.acquire() as conn:
        await conn.execute("INSERT INTO messages (telegram_id, role, content) VALUES ($1, $2, $3)", telegram_id, role, content[:12000])


async def get_chat_history(telegram_id, limit=10):
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
            SET plan=$2, plan_until=$3, daily_used=0, weekly_used=0
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
    current_model = model_names.get(user["selected_model"], "ChatGPT")

    if plan == "VIP":
        usage_block = "Запросов: ♾ безлимит"
    elif plan in {"PLUS", "PRO"}:
        usage_block = f"Запросов в неделю: {user['weekly_used']}/{PLAN_WEEKLY_LIMITS[plan]}"
    else:
        usage_block = (
            f"Запросов сегодня: {user['daily_used']}/{FREE_DAILY_LIMIT}\n"
            f"Запросов в неделю: {user['weekly_used']}/{FREE_WEEKLY_LIMIT}"
        )

    text = (
        f"📊 Статистика использования\n\n"
        f"{usage_block}\n\n"
        f"Подписка: {plan}\n"
        f"Выбрана модель: {current_model}\n\n"
    )

    if plan != "FREE" and user["plan_until"]:
        text += f"Активна до: {user['plan_until'].strftime('%d.%m.%Y')}\n\n"

    text += (
        "Нужно больше? 🚀 Выберите тариф для покупки Premium:\n\n"
        "⭐ PLUS — 500 запросов в неделю\n"
        "💎 PRO — 1400 запросов в неделю\n"
        "👑 VIP — безлимит"
    )
    return text


def normalize_anthropic_messages(messages: list[dict]):
    result = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "user":
            result.append({"role": "user", "content": content})
        elif role == "assistant":
            result.append({"role": "assistant", "content": content})
    if not result:
        result.append({"role": "user", "content": "Привет"})
    return result


def messages_to_plain_text(messages: list[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def ai_router(selected_model: str, messages: list[dict]):
    current_datetime = datetime.now().strftime("%d.%m.%Y %H:%M")
    today_text = datetime.now().strftime("%A, %d.%m.%Y")
    system_text = (
        "Ты профессиональный AI-ассистент. "
        "Отвечай понятно, структурно и по делу. "
        "Если пользователь пишет на русском — отвечай на русском. "
        f"Текущая дата и время: {current_datetime}. Сегодня: {today_text}."
    )

    if selected_model == "claude":
        if not anthropic_client:
            return "⚠️ Claude пока не подключён. Администратору нужно добавить ANTHROPIC_API_KEY в Railway."

        response = await anthropic_client.messages.create(
            model=ANTHROPIC_TEXT_MODEL,
            max_tokens=2500,
            temperature=0.7,
            system=system_text,
            messages=normalize_anthropic_messages(messages),
        )
        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()

    if selected_model == "gemini":
        if not google_client:
            return "⚠️ Gemini пока не подключён. Администратору нужно добавить GOOGLE_API_KEY в Railway."

        prompt = f"{system_text}\n\nИстория диалога:\n{messages_to_plain_text(messages)}"

        def run_gemini():
            response = google_client.models.generate_content(
                model=GEMINI_TEXT_MODEL,
                contents=prompt,
            )
            return getattr(response, "text", "") or ""

        return (await asyncio.to_thread(run_gemini)).strip()

    if selected_model == "deepseek" and deepseek_client:
        full_messages = [{"role": "system", "content": system_text}, *messages]
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=full_messages,
            temperature=0.7,
        )
        return response.choices[0].message.content

    full_messages = [{"role": "system", "content": system_text}, *messages]
    response = await client.chat.completions.create(
        model=OPENAI_TEXT_MODEL,
        messages=full_messages,
        temperature=0.7,
    )
    return response.choices[0].message.content


async def generate_nano_banana_image(prompt: str) -> tuple[bytes | None, str]:
    if not google_client:
        return None, "⚠️ Nano Banana пока не подключён. Администратору нужно добавить GOOGLE_API_KEY в Railway."

    def run_image_generation():
        response = google_client.models.generate_images(
            model=NANO_BANANA_MODEL,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="1:1",
                person_generation="allow_adult",
            ),
        )

        if not response.generated_images:
            return None, "⚠️ Nano Banana не вернул изображение. Попробуйте другой запрос."

        image_obj = response.generated_images[0].image

        if hasattr(image_obj, "image_bytes") and image_obj.image_bytes:
            return image_obj.image_bytes, "🍌 Готово"

        buffer = BytesIO()
        image_obj.save(buffer, format="PNG")
        return buffer.getvalue(), "🍌 Готово"

    return await asyncio.to_thread(run_image_generation)


async def generate_gpt_image(prompt: str) -> tuple[bytes | None, str]:
    def normalize_b64(value: str) -> bytes:
        if value.startswith("data:image"):
            value = value.split(",", 1)[1]
        return base64.b64decode(value)

    response = await client.images.generate(
        model=GPT_IMAGE_MODEL,
        prompt=prompt,
        size="1024x1024",
        quality="medium",
        n=1,
    )

    item = response.data[0]
    if getattr(item, "b64_json", None):
        return normalize_b64(item.b64_json), "🌀 Готово"

    return None, "⚠️ Sora GPT Image не вернул изображение. Попробуйте другой запрос."


def short_error_text(error: Exception) -> str:
    return str(error).replace("\n", " ")[:1200]


async def send_ai_error_to_admin(error_text: str):
    for admin_id in ADMIN_IDS:
        try:
            await bot.send_message(admin_id, error_text[:3000])
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


def get_tribute_slug(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


def tribute_slug_map():
    result = {}
    for plan, months_map in TRIBUTE_LINKS.items():
        for months, url in months_map.items():
            result[get_tribute_slug(url)] = (plan, months)
    return result


def collect_values(obj):
    values = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            values.append((str(key), value))
            values.extend(collect_values(value))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(collect_values(item))
    return values


def find_telegram_id_in_payload(payload) -> int | None:
    keys = {
        "telegram_id", "telegramid", "tg_id", "tgid", "telegram_user_id",
        "telegramuserid", "telegram_userid", "buyer_telegram_id",
        "customer_telegram_id", "user_telegram_id",
    }
    for key, value in collect_values(payload):
        normalized_key = key.lower().replace("-", "_")
        compact_key = normalized_key.replace("_", "")
        if normalized_key in keys or compact_key in keys:
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
    return None


def find_amount_in_payload(payload) -> int:
    amount_keys = {"amount", "price", "total", "sum", "total_amount", "paid_amount"}
    for key, value in collect_values(payload):
        normalized_key = key.lower().replace("-", "_")
        if normalized_key in amount_keys:
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(round(value))
            if isinstance(value, str):
                cleaned = value.replace(" ", "").replace("₽", "").replace("руб", "").replace(",", ".")
                try:
                    return int(round(float(cleaned)))
                except Exception:
                    pass
    return 0


def find_external_payment_id(payload) -> str:
    id_keys = {"id", "payment_id", "transaction_id", "order_id", "invoice_id", "subscription_id"}
    for key, value in collect_values(payload):
        normalized_key = key.lower().replace("-", "_")
        if normalized_key in id_keys and value:
            return str(value)[:200]
    return ""


def is_tribute_payment_success(payload) -> bool:
    text = json.dumps(payload, ensure_ascii=False).lower()
    if "test" in text and "payment" not in text:
        return False
    success_words = [
        "paid", "payment_succeeded", "payment.succeeded", "successful",
        "success", "completed", "confirmed", "approved", "оплачен", "оплата",
    ]
    cancel_words = ["failed", "cancel", "refunded", "refund", "declined", "expired"]
    if any(word in text for word in cancel_words):
        return False
    return any(word in text for word in success_words)


def find_plan_months_in_payload(payload) -> tuple[str | None, int | None]:
    text = json.dumps(payload, ensure_ascii=False)
    lower_text = text.lower()

    for slug, plan_months in tribute_slug_map().items():
        if slug.lower() in lower_text:
            return plan_months

    plan = None
    if "безлимит" in lower_text or "vip" in lower_text:
        plan = "VIP"
    elif "1400" in lower_text or "pro" in lower_text:
        plan = "PRO"
    elif "500" in lower_text or "plus" in lower_text:
        plan = "PLUS"

    months = None
    for value in (12, 6, 3, 1):
        if f"{value} месяц" in lower_text or f"{value} мес" in lower_text or f"{value}m" in lower_text:
            months = value
            break

    return plan, months


async def create_tribute_session(telegram_id: int, plan: str, months: int) -> str:
    token = secrets.token_urlsafe(24)
    tribute_url = TRIBUTE_LINKS[plan][months]
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO tribute_sessions (token, telegram_id, plan, months, tribute_url)
            VALUES ($1, $2, $3, $4, $5)
            """,
            token,
            telegram_id,
            plan,
            months,
            tribute_url,
        )
    return f"{PUBLIC_BASE_URL}/start-buy?token={token}"


async def find_pending_tribute_session(plan: str | None, months: int | None):
    if not plan or not months:
        return None

    async with db_pool.acquire() as conn:
        session = await conn.fetchrow(
            """
            SELECT * FROM tribute_sessions
            WHERE plan=$1
              AND months=$2
              AND status='clicked'
              AND clicked_at > NOW() - INTERVAL '45 minutes'
            ORDER BY clicked_at DESC
            LIMIT 1
            """,
            plan,
            months,
        )

        if session:
            return session

        session = await conn.fetchrow(
            """
            SELECT * FROM tribute_sessions
            WHERE plan=$1
              AND months=$2
              AND status='created'
              AND created_at > NOW() - INTERVAL '45 minutes'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            plan,
            months,
        )

    return session


async def record_tribute_payment_and_activate(telegram_id: int, plan: str, months: int, amount: int, payload: dict, external_id: str = ""):
    raw_payload = json.dumps(payload, ensure_ascii=False)[:12000]
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO payments (
                telegram_id, plan, months, amount, currency, payload,
                telegram_payment_charge_id, provider_payment_charge_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            telegram_id,
            plan,
            months,
            amount,
            "RUB",
            raw_payload,
            external_id,
            external_id,
        )
        await conn.execute(
            """
            UPDATE tribute_sessions
            SET status='paid'
            WHERE telegram_id=$1 AND plan=$2 AND months=$3 AND status IN ('created', 'clicked')
            """,
            telegram_id,
            plan,
            months,
        )

    await activate_plan(telegram_id, plan, months)

    try:
        await bot.send_message(
            telegram_id,
            f"✅ Оплата через банковскую карту / СБП прошла успешно!\n\nТариф {plan} активирован на {months} мес.",
            reply_markup=main_menu(),
        )
    except Exception as e:
        print(f"TRIBUTE USER NOTIFY ERROR: {e}")

    for admin_id in ADMIN_IDS:
        try:
            await bot.send_message(
                admin_id,
                f"💳 Tribute оплата активирована\n\nID: {telegram_id}\nТариф: {plan}\nПериод: {months} мес.\nСумма: {amount} RUB",
            )
        except Exception:
            pass


async def handle_start_buy(request: web.Request):
    token = request.query.get("token", "")
    if not token:
        return web.Response(text="Payment token is missing", status=400)

    async with db_pool.acquire() as conn:
        session = await conn.fetchrow("SELECT * FROM tribute_sessions WHERE token=$1", token)
        if not session:
            return web.Response(text="Payment session not found", status=404)
        await conn.execute("UPDATE tribute_sessions SET status='clicked', clicked_at=NOW() WHERE token=$1", token)

    raise web.HTTPFound(session["tribute_url"])


async def handle_tribute_webhook(request: web.Request):
    try:
        try:
            payload = await request.json()
        except Exception:
            raw_text = await request.text()
            payload = {"raw": raw_text}

        await log_event(None, "tribute_webhook_raw", json.dumps(payload, ensure_ascii=False)[:1000])

        if not is_tribute_payment_success(payload):
            return web.json_response({"ok": True, "status": "ignored_non_success_event"})

        plan, months = find_plan_months_in_payload(payload)
        telegram_id = find_telegram_id_in_payload(payload)

        if not telegram_id:
            session = await find_pending_tribute_session(plan, months)
            if session:
                telegram_id = session["telegram_id"]
                plan = session["plan"]
                months = session["months"]

        if not telegram_id or not plan or not months or plan not in TARIFFS:
            details = json.dumps(payload, ensure_ascii=False)[:2500]
            await log_event(None, "tribute_webhook_unmatched", details)
            for admin_id in ADMIN_IDS:
                try:
                    await bot.send_message(
                        admin_id,
                        "⚠️ Tribute webhook пришёл, но не удалось понять пользователя/тариф.\n\n"
                        f"plan={plan}, months={months}, telegram_id={telegram_id}\n\n"
                        f"payload: {details[:1500]}",
                    )
                except Exception:
                    pass
            return web.json_response({"ok": False, "error": "cannot_match_payment"}, status=202)

        amount = find_amount_in_payload(payload)
        external_id = find_external_payment_id(payload)
        await record_tribute_payment_and_activate(telegram_id, plan, months, amount, payload, external_id)
        return web.json_response({"ok": True, "activated": True, "telegram_id": telegram_id, "plan": plan, "months": months})

    except Exception as e:
        print(f"TRIBUTE WEBHOOK ERROR: {e}")
        print(traceback.format_exc())
        return web.json_response({"ok": False, "error": short_error_text(e)}, status=500)


async def handle_health(request: web.Request):
    return web.json_response({"ok": True, "service": "gptclaude-bot"})


async def start_web_server():
    app = web.Application()
    app.router.add_get("/", handle_health)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/start-buy", handle_start_buy)
    app.router.add_post("/tribute-webhook", handle_tribute_webhook)
    app.router.add_get("/tribute-webhook", handle_health)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    print(f"WEB SERVER STARTED ON PORT {PORT}")
    return runner


@dp.message(CommandStart())
async def start_handler(message: Message):
    now = time.time()
    last = recent_starts.get(message.from_user.id, 0)
    if now - last < 2:
        return
    recent_starts[message.from_user.id] = now

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


@dp.message(Command("premium"))
async def premium_command(message: Message):
    await log_event(message.from_user.id, "premium_open")
    await message.answer(premium_text(), reply_markup=tariffs_menu())


@dp.message(Command("models"))
async def models_command(message: Message):
    await log_event(message.from_user.id, "models_command")
    await message.answer("🤖 Выберите нейросеть:", reply_markup=models_menu())


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
    await safe_edit_or_send(callback, "🤖 Выберите нейросеть:", reply_markup=models_menu())


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


@dp.callback_query(F.data == "rub_payment_disabled")
async def rub_payment_disabled_callback(callback: CallbackQuery):
    await callback.answer("Оплата рублями скоро будет доступна", show_alert=True)
    await log_event(callback.from_user.id, "rub_payment_disabled_click")


@dp.callback_query(F.data.startswith("pay_tribute_"))
async def pay_tribute_callback(callback: CallbackQuery):
    await callback.answer()
    _, _, plan, months = callback.data.split("_")
    months = int(months)
    if plan not in TARIFFS or months not in TRIBUTE_LINKS[plan]:
        await safe_edit_or_send(callback, "⚠️ Этот способ оплаты сейчас недоступен.", reply_markup=tariffs_menu())
        return

    await get_or_create_user_by_data(callback.from_user.id, callback.from_user.username, callback.from_user.first_name)
    payment_url = await create_tribute_session(callback.from_user.id, plan, months)
    await log_event(callback.from_user.id, "tribute_payment_open", f"{plan} {months}")

    await safe_edit_or_send(
        callback,
        f"💳 Оплата банковской картой / СБП\n\nТариф: {plan}\nПериод: {months} мес.\n\nПосле оплаты бот автоматически активирует подписку.",
        reply_markup=tribute_open_payment_menu(payment_url, plan, months),
    )


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
    await message.answer(f"✅ Оплата прошла успешно!\n\nТариф {plan} активирован на {months} мес.", reply_markup=main_menu())


@dp.callback_query(F.data.startswith("set_model_"))
async def set_model_callback(callback: CallbackQuery):
    await callback.answer()
    model = callback.data.replace("set_model_", "")
    allowed_models = {"gpt", "claude", "gemini", "nanobanana", "gptimage"}

    if model not in allowed_models:
        await safe_edit_or_send(callback, "⚠️ Эта модель сейчас недоступна.", reply_markup=models_menu())
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
        text=f"✅ Нейросеть выбрана:\n\n{model_display_name(model)}\n\nНапишите запрос или выберите действие ниже.",
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
        "/setplus telegram_id\n"
        "/setpro telegram_id\n"
        "/setvip telegram_id\n"
        "/setfree telegram_id"
    )


@dp.message(Command("stats"))
async def stats_handler(message: Message):
    if not is_admin(message.from_user.id):
        return

    async with db_pool.acquire() as conn:
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        today_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE created_at::date = CURRENT_DATE")
        total_messages = await conn.fetchval("SELECT COUNT(*) FROM messages WHERE role='user'")
        today_messages = await conn.fetchval("SELECT COUNT(*) FROM messages WHERE role='user' AND created_at::date = CURRENT_DATE")
        plus_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE plan='PLUS'")
        pro_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE plan='PRO'")
        vip_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE plan='VIP'")
        total_stars = await conn.fetchval("SELECT COALESCE(SUM(amount), 0) FROM payments")
        starts_today = await conn.fetchval("SELECT COUNT(*) FROM events WHERE event_type='start' AND created_at::date = CURRENT_DATE")
        premium_clicks = await conn.fetchval("SELECT COUNT(*) FROM events WHERE event_type='premium_click'")
        invoices = await conn.fetchval("SELECT COUNT(*) FROM events WHERE event_type='invoice_open'")

    await message.answer(
        "📊 Статистика бота\n\n"
        f"Пользователей всего: {total_users}\n"
        f"Новых сегодня: {today_users}\n"
        f"Стартов сегодня: {starts_today}\n\n"
        f"Сообщений всего: {total_messages}\n"
        f"Сообщений сегодня: {today_messages}\n\n"
        f"PLUS: {plus_users}\n"
        f"PRO: {pro_users}\n"
        f"VIP: {vip_users}\n\n"
        f"Открытий премиума: {premium_clicks}\n"
        f"Открытий оплаты Stars: {invoices}\n"
        f"Stars получено: {total_stars}"
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

    if selected_model in {"nanobanana", "gptimage"}:
        wait_text = "🍌 Генерирую изображение..." if selected_model == "nanobanana" else "🌀 Генерирую изображение..."
        wait_message = await message.answer(wait_text)
        try:
            await save_message(message.from_user.id, "user", message.text)

            if selected_model == "nanobanana":
                image_bytes, text_note = await generate_nano_banana_image(message.text)
            else:
                image_bytes, text_note = await generate_gpt_image(message.text)

            if not image_bytes:
                await wait_message.edit_text(
                    text_note or "⚠️ Генерация не вернула изображение. Попробуйте другой запрос.",
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
            await log_event(message.from_user.id, "ai_image", selected_model)
            return

        except Exception as e:
            admin_error = short_error_text(e)
            print(f"IMAGE ERROR SHORT:\n{admin_error}")
            print(f"IMAGE ERROR TRACE:\n{traceback.format_exc()}")
            await send_ai_error_to_admin(f"⚠️ AI ERROR | {selected_model} | {admin_error}")
            try:
                await wait_message.edit_text(
                    "⚠️ Генерация изображений временно недоступна.\n\n"
                    "Попробуйте позже или выберите другую нейросеть.",
                    reply_markup=main_menu(),
                )
            except Exception:
                await message.answer(
                    "⚠️ Генерация изображений временно недоступна.\n\n"
                    "Попробуйте позже или выберите другую нейросеть.",
                    reply_markup=main_menu(),
                )
            return

    wait_message = await message.answer("Печатает ответ...")

    try:
        await save_message(message.from_user.id, "user", message.text)
        history = await get_chat_history(message.from_user.id)
        answer = await ai_router(selected_model, history)

        if not answer:
            answer = "⚠️ AI вернул пустой ответ. Попробуйте переформулировать вопрос."

        await save_message(message.from_user.id, "assistant", answer)
        await increase_usage(message.from_user.id)
        await log_event(message.from_user.id, "ai_message", selected_model)

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
