import asyncio
import os
import time
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
)
from openai import AsyncOpenAI


BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

ADMIN_IDS = {
    int(x.strip())
    for x in os.getenv("ADMIN_IDS", "").split(",")
    if x.strip().isdigit()
}

FREE_DAILY_LIMIT = 15
FREE_WEEKLY_LIMIT = 105
PRO_DAILY_LIMIT = 1000

PRO_STARS_PRICE = int(os.getenv("PRO_STARS_PRICE", "299"))
VIP_STARS_PRICE = int(os.getenv("VIP_STARS_PRICE", "1499"))

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


menu = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="💬 Новый чат", callback_data="new_chat"),
            InlineKeyboardButton(text="👤 Профиль", callback_data="profile"),
        ],
        [
            InlineKeyboardButton(text="💎 Тарифы", callback_data="plans"),
            InlineKeyboardButton(text="🤖 Модель", callback_data="models"),
        ],
    ]
)

plans_menu = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text=f"💎 PRO — {PRO_STARS_PRICE} ⭐", callback_data="buy_pro")],
        [InlineKeyboardButton(text=f"👑 VIP — {VIP_STARS_PRICE} ⭐", callback_data="buy_vip")],
        [InlineKeyboardButton(text="⬅️ Назад", callback_data="profile")],
    ]
)

models_menu = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="🧠 ChatGPT 5", callback_data="set_model_gpt")],
        [InlineKeyboardButton(text="🟣 Claude", callback_data="set_model_claude")],
        [InlineKeyboardButton(text="🔵 Gemini", callback_data="set_model_gemini")],
        [InlineKeyboardButton(text="⚫ DeepSeek", callback_data="set_model_deepseek")],
    ]
)


def get_week_start():
    today = date.today()
    return today - timedelta(days=today.weekday())


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


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

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS spam_state (
                telegram_id BIGINT PRIMARY KEY,
                window_start BIGINT DEFAULT 0,
                message_count INTEGER DEFAULT 0,
                blocked_until BIGINT DEFAULT 0
            );
        """)


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
        state = await conn.fetchrow(
            "SELECT * FROM spam_state WHERE telegram_id=$1",
            telegram_id,
        )

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

    if plan == "PRO":
        if user["daily_used"] >= PRO_DAILY_LIMIT:
            return False, "PRO_DAY_LIMIT"
        return True, None

    if user["daily_used"] >= FREE_DAILY_LIMIT:
        return False, "FREE_DAY_LIMIT"

    if user["weekly_used"] >= FREE_WEEKLY_LIMIT:
        return False, "FREE_WEEK_LIMIT"

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
    content = content[:12000]
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO messages (telegram_id, role, content)
            VALUES ($1, $2, $3)
        """, telegram_id, role, content)


async def get_chat_history(telegram_id, limit=12):
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT role, content
            FROM messages
            WHERE telegram_id=$1
            ORDER BY created_at DESC
            LIMIT $2
        """, telegram_id, limit)

    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


async def clear_chat(telegram_id):
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM messages WHERE telegram_id=$1", telegram_id)


async def set_user_plan(telegram_id: int, plan: str):
    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE users
            SET plan=$2
            WHERE telegram_id=$1
        """, telegram_id, plan)


async def user_profile_text(user):
    model_names = {
        "gpt": "ChatGPT 5",
        "claude": "Claude",
        "gemini": "Gemini",
        "deepseek": "DeepSeek",
    }

    plan = user["plan"]
    current_model = model_names.get(user["selected_model"], "ChatGPT 5")

    if plan == "VIP":
        limit_text = "♾ Безлимит"
    elif plan == "PRO":
        limit_text = f"{user['daily_used']} / {PRO_DAILY_LIMIT} сегодня"
    else:
        limit_text = (
            f"{user['daily_used']} / {FREE_DAILY_LIMIT} сегодня\n"
            f"{user['weekly_used']} / {FREE_WEEKLY_LIMIT} за неделю"
        )

    return (
        "👤 Ваш профиль\n\n"
        f"Тариф: {plan}\n"
        f"Модель: {current_model}\n\n"
        f"Использование:\n{limit_text}"
    )


async def send_long_text(message_or_callback_message, text: str, reply_markup=None):
    chunks = [text[i:i + 3900] for i in range(0, len(text), 3900)]
    for i, chunk in enumerate(chunks):
        await message_or_callback_message.answer(
            chunk,
            reply_markup=reply_markup if i == len(chunks) - 1 else None,
        )


async def ai_router(selected_model: str, messages: list[dict]):
    system = {
        "role": "system",
        "content": (
            "Ты профессиональный AI-ассистент. "
            "Отвечай понятно, структурно и по делу. "
            "Если пользователь пишет на русском — отвечай на русском."
        ),
    }

    full_messages = [system, *messages]

    if selected_model == "deepseek" and deepseek_client:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=full_messages,
            temperature=0.7,
        )
        return response.choices[0].message.content

    if selected_model == "claude":
        if not ANTHROPIC_API_KEY:
            return await openai_answer(full_messages, "gpt-4o-mini")
        return await openai_answer(full_messages, "gpt-4o-mini")

    if selected_model == "gemini":
        if not GOOGLE_API_KEY:
            return await openai_answer(full_messages, "gpt-4o-mini")
        return await openai_answer(full_messages, "gpt-4o-mini")

    return await openai_answer(full_messages, "gpt-4o-mini")


async def openai_answer(messages: list[dict], model: str):
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content


@dp.message(CommandStart())
async def start_handler(message: Message):
    await get_or_create_user(message)

    await message.answer(
        "🚀 Добро пожаловать в ChatGPT + Claude + Gemini + DeepSeek!\n\n"
        "Напишите любой вопрос, и я отвечу с помощью AI.",
        reply_markup=menu,
    )


@dp.message(Command("admin"))
async def admin_handler(message: Message):
    if not is_admin(message.from_user.id):
        return

    async with db_pool.acquire() as conn:
        total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
        pro_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE plan='PRO'")
        vip_users = await conn.fetchval("SELECT COUNT(*) FROM users WHERE plan='VIP'")
        total_messages = await conn.fetchval("SELECT COUNT(*) FROM messages")
        today_users = await conn.fetchval("""
            SELECT COUNT(*) FROM users
            WHERE created_at::date = CURRENT_DATE
        """)

    await message.answer(
        "🛠 Админка\n\n"
        f"Пользователей всего: {total_users}\n"
        f"Новых сегодня: {today_users}\n"
        f"PRO: {pro_users}\n"
        f"VIP: {vip_users}\n"
        f"Сообщений в памяти: {total_messages}\n\n"
        "Команды:\n"
        "/setpro telegram_id\n"
        "/setvip telegram_id\n"
        "/setfree telegram_id\n"
        "/stats"
    )


@dp.message(Command("stats"))
async def stats_handler(message: Message):
    if not is_admin(message.from_user.id):
        return

    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT plan, COUNT(*) AS count
            FROM users
            GROUP BY plan
            ORDER BY plan
        """)

    text = "📊 Статистика тарифов\n\n"
    for row in rows:
        text += f"{row['plan']}: {row['count']}\n"

    await message.answer(text)


async def admin_set_plan(message: Message, plan: str):
    if not is_admin(message.from_user.id):
        return

    parts = message.text.split()
    if len(parts) != 2 or not parts[1].isdigit():
        await message.answer(f"Формат: /set{plan.lower()} telegram_id")
        return

    telegram_id = int(parts[1])
    async with db_pool.acquire() as conn:
        exists = await conn.fetchval(
            "SELECT COUNT(*) FROM users WHERE telegram_id=$1",
            telegram_id,
        )

    if not exists:
        await message.answer("Пользователь не найден.")
        return

    await set_user_plan(telegram_id, plan)
    await message.answer(f"✅ Пользователю {telegram_id} установлен тариф {plan}.")


@dp.message(Command("setpro"))
async def setpro_handler(message: Message):
    await admin_set_plan(message, "PRO")


@dp.message(Command("setvip"))
async def setvip_handler(message: Message):
    await admin_set_plan(message, "VIP")


@dp.message(Command("setfree"))
async def setfree_handler(message: Message):
    await admin_set_plan(message, "FREE")


@dp.callback_query(F.data == "profile")
async def profile_callback(callback: CallbackQuery):
    user = await get_or_create_user_by_data(
        telegram_id=callback.from_user.id,
        username=callback.from_user.username,
        first_name=callback.from_user.first_name,
    )

    await callback.message.answer(await user_profile_text(user), reply_markup=menu)
    await callback.answer()


@dp.callback_query(F.data == "plans")
async def plans_callback(callback: CallbackQuery):
    await callback.message.answer(
        "💎 Тарифы\n\n"
        "FREE — 15 сообщений в день / 105 в неделю\n"
        f"PRO — {PRO_DAILY_LIMIT} сообщений в день\n"
        "VIP — безлимит\n\n"
        "Оплата доступна через Telegram Stars.",
        reply_markup=plans_menu,
    )
    await callback.answer()


@dp.callback_query(F.data.in_({"buy_pro", "buy_vip"}))
async def buy_plan_callback(callback: CallbackQuery):
    plan = "PRO" if callback.data == "buy_pro" else "VIP"
    price = PRO_STARS_PRICE if plan == "PRO" else VIP_STARS_PRICE
    payload = f"plan:{plan}:user:{callback.from_user.id}:ts:{int(time.time())}"

    await bot.send_invoice(
        chat_id=callback.message.chat.id,
        title=f"Тариф {plan}",
        description=f"Активация тарифа {plan} для AI-бота.",
        payload=payload,
        provider_token="",
        currency="XTR",
        prices=[LabeledPrice(label=f"Тариф {plan}", amount=price)],
    )

    await callback.answer()


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
    plan = "PRO"

    if len(parts) >= 2 and parts[0] == "plan":
        plan = parts[1]

    if plan not in {"PRO", "VIP"}:
        await message.answer("⚠️ Платёж получен, но тариф не распознан. Напишите администратору.")
        return

    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO payments (
                telegram_id, plan, amount, currency, payload,
                telegram_payment_charge_id, provider_payment_charge_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            message.from_user.id,
            plan,
            payment.total_amount,
            payment.currency,
            payload,
            payment.telegram_payment_charge_id,
            payment.provider_payment_charge_id,
        )

        await conn.execute("""
            UPDATE users
            SET plan=$2
            WHERE telegram_id=$1
        """, message.from_user.id, plan)

    await message.answer(
        f"✅ Оплата прошла успешно!\n\n"
        f"Тариф {plan} активирован.",
        reply_markup=menu,
    )


@dp.callback_query(F.data == "models")
async def models_callback(callback: CallbackQuery):
    await callback.message.answer("🤖 Выберите модель:", reply_markup=models_menu)
    await callback.answer()


@dp.callback_query(F.data.startswith("set_model_"))
async def set_model_callback(callback: CallbackQuery):
    model = callback.data.replace("set_model_", "")

    async with db_pool.acquire() as conn:
        await conn.execute("""
            UPDATE users
            SET selected_model=$1
            WHERE telegram_id=$2
        """, model, callback.from_user.id)

    names = {
        "gpt": "🧠 ChatGPT 5",
        "claude": "🟣 Claude",
        "gemini": "🔵 Gemini",
        "deepseek": "⚫ DeepSeek",
    }

    await callback.message.answer(
        f"✅ Модель переключена:\n\n{names.get(model)}",
        reply_markup=menu,
    )

    await callback.answer()


@dp.callback_query(F.data == "new_chat")
async def new_chat_callback(callback: CallbackQuery):
    await clear_chat(callback.from_user.id)

    await callback.message.answer(
        "💬 Новый чат начат.\n\nКонтекст очищен.",
        reply_markup=menu,
    )

    await callback.answer()


@dp.message(F.text)
async def chat_handler(message: Message):
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
        if reason == "FREE_DAY_LIMIT":
            text = (
                "⏳ Дневной лимит FREE на сегодня закончился.\n\n"
                "Завтра сообщения обновятся автоматически.\n\n"
                "Можно перейти на PRO в разделе «Тарифы»."
            )
        elif reason == "FREE_WEEK_LIMIT":
            text = (
                "⏳ Недельный лимит FREE закончился.\n\n"
                "Лимит обновится на следующей неделе.\n\n"
                "Можно перейти на PRO в разделе «Тарифы»."
            )
        else:
            text = "⏳ Дневной лимит PRO на сегодня закончился."

        await message.answer(text, reply_markup=plans_menu)
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

        if len(answer) <= 3900:
            await wait_message.edit_text(answer)
        else:
            await wait_message.edit_text(answer[:3900])
            for chunk in [answer[i:i + 3900] for i in range(3900, len(answer), 3900)]:
                await message.answer(chunk)

    except Exception as e:
        print(f"AI ERROR: {e}")

        await wait_message.edit_text(
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

    await init_db()

    print("DATABASE CONNECTED")
    print("BOT STARTED")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
