import asyncio
import os
import time
import traceback
from datetime import date, datetime, timedelta

import asyncpg
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
)
from openai import AsyncOpenAI


BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

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

SPAM_WINDOW_SECONDS = 8
SPAM_MAX_MESSAGES = 5
SPAM_BLOCK_SECONDS = 60

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
) if DEEPSEEK_API_KEY else None

db_pool = None
recent_starts = {}


def main_menu():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👤 Профиль", callback_data="profile"),
                InlineKeyboardButton(text="💳 Купить подписку", callback_data="premium"),
            ],
            [
                InlineKeyboardButton(text="🤖 Модель", callback_data="models"),
            ],
            [
                InlineKeyboardButton(text="🧠 Наши каналы", callback_data="channels"),
            ],
        ]
    )


def models_menu():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🧠 ChatGPT", callback_data="set_model_gpt")],
            [InlineKeyboardButton(text="🟣 Claude", callback_data="set_model_claude")],
            [InlineKeyboardButton(text="🍌 Nano Banana", callback_data="set_model_nanobanana")],
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
            [InlineKeyboardButton(text="⭐ Оплатить Telegram Stars", callback_data=f"pay_stars_{plan}_{months}")],
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


def welcome_text():
    return (
        "👋 Добро пожаловать в @GPTclaudeAIbot\n\n"
        "Ваш AI-бот для работы с нейросетями в одном месте.\n\n"
        "📝 Генерация текста:\n"
        "• ChatGPT\n"
        "• Claude\n\n"
        "🌇 Генерация изображений:\n"
        "• Nano Banana Pro\n\n"
        "🧠 Наши каналы:\n"
        "• Наш канал: <a href='https://t.me/ToporLive1_0'>Топор Live 1.0</a>\n"
        "• Канал support: <a href='https://t.me/LightningNews'>Молния News</a>\n\n"
        "Напишите вопрос или выберите действие ниже."
    )


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
            await conn.execute("""
                INSERT INTO events (telegram_id, event_type, details)
                VALUES ($1, $2, $3)
            """, telegram_id, event_type, details[:1000])
    except Exception as e:
        print(f"LOG EVENT ERROR: {e}")


async def setup_bot_info():
    await bot.set_my_commands([
        BotCommand(command="start", description="👋 Что умеет бот"),
        BotCommand(command="account", description="👤 Мой профиль"),
        BotCommand(command="premium", description="🚀 Премиум"),
        BotCommand(command="deletecontext", description="💬 Удалить контекст"),
    ])

    try:
        await bot.set_my_description(
            "AI-бот для работы с ChatGPT, Claude, Gemini и DeepSeek."
        )
        await bot.set_my_short_description(
            "ChatGPT, Claude, Gemini и DeepSeek в Telegram"
        )
    except Exception as e:
        print(f"BOT DESCRIPTION ERROR: {e}")


async def get_or_create_user_by_data(telegram_id, username=None, first_name=None):
    today = date.today()
    week_start = get_week_start()

    async with db_pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM users WHERE telegram_id=$1", telegram_id)

        if not user:
            await conn.execute("""
                INSERT INTO users 
                (telegram_id, username, first_name, day_start, week_start)
                VALUES ($1, $2, $3, $4, $5)
            """, telegram_id, username, first_name, today, week_start)
            await log_event(telegram_id, "new_user", username or "")

        else:
            await conn.execute("""
                UPDATE users
                SET username=$2, first_name=$3
                WHERE telegram_id=$1
            """, telegram_id, username, first_name)

        user = await conn.fetchrow("SELECT * FROM users WHERE telegram_id=$1", telegram_id)

        if user["day_start"] != today:
            await conn.execute("""
                UPDATE users
                SET daily_used=0, day_start=$2
                WHERE telegram_id=$1
            """, telegram_id, today)

        if user["week_start"] != week_start:
            await conn.execute("""
                UPDATE users
                SET weekly_used=0, week_start=$2
                WHERE telegram_id=$1
            """, telegram_id, week_start)

        user = await conn.fetchrow("SELECT * FROM users WHERE telegram_id=$1", telegram_id)

        if user["plan"] != "FREE" and user["plan_until"] and user["plan_until"] < datetime.utcnow():
            await conn.execute("""
                UPDATE users
                SET plan='FREE', plan_until=NULL
                WHERE telegram_id=$1
            """, telegram_id)
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
            await conn.execute("""
                UPDATE spam_state
                SET window_start=$2, message_count=1, blocked_until=0
                WHERE telegram_id=$1
            """, telegram_id, now)
            return True, None

        new_count = state["message_count"] + 1

        if new_count > SPAM_MAX_MESSAGES:
            blocked_until = now + SPAM_BLOCK_SECONDS
            await conn.execute("""
                UPDATE spam_state
                SET message_count=$2, blocked_until=$3
                WHERE telegram_id=$1
            """, telegram_id, new_count, blocked_until)
            await log_event(telegram_id, "spam_block", str(SPAM_BLOCK_SECONDS))
            return False, SPAM_BLOCK_SECONDS

        await conn.execute("""
            UPDATE spam_state
            SET message_count=$2
            WHERE telegram_id=$1
        """, telegram_id, new_count)

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
        await conn.execute("""
            INSERT INTO messages (telegram_id, role, content)
            VALUES ($1, $2, $3)
        """, telegram_id, role, content[:12000])


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
    "nanobanana": "Nano Banana",
}

    plan = user["plan"] or "FREE"
    current_model = model_names.get(user["selected_model"], "ChatGPT 5")

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
    "Нужно больше? 🚀 Выберите тариф для покупки:\n\n"
    "⭐ PLUS — 500 запросов в неделю\n"
    "💎 PRO — 1400 запросов в неделю\n"
    "👑 VIP — безлимит"
)

    return text


async def ai_router(selected_model: str, messages: list[dict]):
    system_message = {
        "role": "system",
        "content": (
            "Ты профессиональный AI-ассистент. "
            "Отвечай понятно, структурно и по делу. "
            "Если пользователь пишет на русском — отвечай на русском."
        ),
    }

    full_messages = [system_message, *messages]

    if selected_model == "deepseek" and deepseek_client:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=full_messages,
            temperature=0.7,
        )
        return response.choices[0].message.content

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=full_messages,
        temperature=0.7,
    )

    return response.choices[0].message.content


async def send_ai_error_to_admin(error_text: str):
    for admin_id in ADMIN_IDS:
        try:
            await bot.send_message(admin_id, f"⚠️ AI ERROR\n\n{error_text[:3500]}")
        except Exception:
            pass


async def safe_edit_or_send(callback: CallbackQuery, text: str, reply_markup=None):
    try:
        await callback.message.edit_text(text, reply_markup=reply_markup)
    except Exception:
        await callback.message.answer(text, reply_markup=reply_markup)


@dp.message(CommandStart())
async def start_handler(message: Message):
    now = time.time()
    last = recent_starts.get(message.from_user.id, 0)

    if now - last < 2:
        return

    recent_starts[message.from_user.id] = now

    await get_or_create_user(message)
    await log_event(message.from_user.id, "start")
    await message.answer(welcome_text(), reply_markup=main_menu(), parse_mode="HTML")


@dp.message(Command("account"))
async def account_command(message: Message):
    user = await get_or_create_user(message)
    await log_event(message.from_user.id, "account")
    await message.answer(await user_profile_text(user), reply_markup=main_menu())


@dp.message(Command("premium"))
async def premium_command(message: Message):
    await log_event(message.from_user.id, "premium_open")
    await message.answer(
"💳 Купить подписку\n\n"
"⭐ PLUS — 500 запросов в неделю\n"
"💎 PRO — 1400 запросов в неделю\n"
"👑 VIP — безлимит",
        reply_markup=tariffs_menu(),
    )


@dp.message(Command("deletecontext"))
async def delete_context_command(message: Message):
    await clear_chat(message.from_user.id)
    await message.answer("💬 Контекст очищен. Новый чат начат.", reply_markup=main_menu())


@dp.callback_query(F.data == "back_main")
async def back_main_callback(callback: CallbackQuery):
    await callback.answer()
    await safe_edit_or_send(callback, welcome_text(), reply_markup=main_menu())


@dp.callback_query(F.data == "profile")
async def profile_callback(callback: CallbackQuery):
    @dp.callback_query(F.data == "channels")
async def channels_callback(callback: CallbackQuery):
    await callback.answer()
    await log_event(callback.from_user.id, "channels_open")

    await safe_edit_or_send(
        callback,
        "🧠 Наши каналы:\n\n"
        "• Наш канал: <a href='https://t.me/ToporLive1_0'>Топор Live 1.0</a>\n"
        "• Канал support: <a href='https://t.me/LightningNews'>Молния News</a>",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="📰 Топор Live 1.0", url="https://t.me/ToporLive1_0")],
                [InlineKeyboardButton(text="⚡ Молния News", url="https://t.me/LightningNews")],
                [InlineKeyboardButton(text="← Назад", callback_data="back_main")],
            ]
        ),
    )    
    await callback.answer("Открываю профиль...")

    user = await get_or_create_user_by_data(
        telegram_id=callback.from_user.id,
        username=callback.from_user.username,
        first_name=callback.from_user.first_name,
    )

    await log_event(callback.from_user.id, "profile_click")
    await callback.message.answer(await user_profile_text(user), reply_markup=main_menu())


@dp.callback_query(F.data.in_({"premium", "plans"}))
async def premium_callback(callback: CallbackQuery):
    await callback.answer()
    await log_event(callback.from_user.id, "premium_click")

    await safe_edit_or_send(
        callback,
"💳 Купить подписку\n\n"
"⭐ PLUS — 500 запросов в неделю\n"
"💎 PRO — 1400 запросов в неделю\n"
"👑 VIP — безлимит",
        reply_markup=tariffs_menu(),
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
        f"🚀 {tariff['title']}\n\n"
        f"{tariff['description']}\n\n"
        "Выберите период подписки:",
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
        f"⭐ Подтвердите оплату Telegram Stars:\n\n"
        f"Тариф: {plan}\n"
        f"Период: {months} мес.\n"
        f"Цена: ⭐ {price}",
        reply_markup=payment_method_menu(plan, months),
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
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
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

    await message.answer(
        f"✅ Оплата прошла успешно!\n\n"
        f"Тариф {plan} активирован на {months} мес.",
        reply_markup=main_menu(),
    )


@dp.callback_query(F.data == "models")
async def models_callback(callback: CallbackQuery):
    await callback.answer()
    await log_event(callback.from_user.id, "models_open")
    await safe_edit_or_send(callback, "🤖 Выберите модель:", reply_markup=models_menu())


@dp.callback_query(F.data.startswith("set_model_"))
async def set_model_callback(callback: CallbackQuery):
    await callback.answer()

    model = callback.data.replace("set_model_", "")

    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE users
            SET selected_model=$1
            WHERE telegram_id=$2
        """, model, callback.from_user.id)

    await log_event(callback.from_user.id, "model_select", model)

 names = {
    "gpt": "🧠 ChatGPT",
    "claude": "🟣 Claude",
    "nanobanana": "🍌 Nano Banana",
}

    await safe_edit_or_send(
        callback,
        f"✅ Модель переключена:\n\n{names.get(model)}",
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
        f"Открытий оплаты: {invoices}\n"
        f"Stars получено: {total_stars}"
    )


@dp.message(Command("users"))
async def users_handler(message: Message):
    if not is_admin(message.from_user.id):
        return

    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT telegram_id, username, first_name, plan, weekly_used, created_at
            FROM users
            ORDER BY created_at DESC
            LIMIT 15
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
            FROM payments
            ORDER BY created_at DESC
            LIMIT 15
        """)

    if not rows:
        await message.answer("Платежей пока нет.")
        return

    text = "💳 Последние платежи\n\n"

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
        await conn.execute("""
            UPDATE users
            SET plan=$2, plan_until=$3
            WHERE telegram_id=$1
        """, telegram_id, plan, until)

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
        await message.answer(
            f"🛡 Слишком много сообщений подряд.\n\n"
            f"Попробуйте снова через {wait_seconds} сек."
        )
        return

    allowed, reason = await check_limit(user)

    if not allowed:
        await log_event(message.from_user.id, "limit_reached", reason)

        await message.answer(
            "⏳ Лимит сообщений закончился.\n\n"
            "Вы можете перейти на PLUS, PRO или VIP.",
            reply_markup=tariffs_menu(),
        )
        return

    wait_message = await message.answer("Печатает ответ...")

    try:
        await save_message(message.from_user.id, "user", message.text)

        history = await get_chat_history(message.from_user.id)
        answer = await ai_router(user["selected_model"], history)

        if not answer:
            answer = "⚠️ AI вернул пустой ответ. Попробуйте переформулировать вопрос."

        await save_message(message.from_user.id, "assistant", answer)
        await increase_usage(message.from_user.id)
        await log_event(message.from_user.id, "ai_message", user["selected_model"])

        if len(answer) <= 3900:
            await wait_message.edit_text(answer)
        else:
            await wait_message.edit_text(answer[:3900])
            for i in range(3900, len(answer), 3900):
                await message.answer(answer[i:i + 3900])

    except Exception:
        error_text = traceback.format_exc()
        print(f"AI ERROR:\n{error_text}")
        await send_ai_error_to_admin(error_text)

        try:
            await wait_message.edit_text(
                "⚠️ Сейчас у AI временные технические работы.\n\n"
                "Попробуйте ещё раз через минуту."
            )
        except Exception:
            await message.answer(
                "⚠️ Сейчас у AI временные технические работы.\n\n"
                "Попробуйте ещё раз через минуту."
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

    print("DATABASE CONNECTED")
    print("BOT STARTED")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
