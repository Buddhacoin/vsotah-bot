import asyncio
import os
from datetime import date, timedelta

import asyncpg
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from openai import AsyncOpenAI


BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

FREE_DAILY_LIMIT = 15
FREE_WEEKLY_LIMIT = 105
PRO_DAILY_LIMIT = 1000

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

db_pool = None


menu = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="💬 Новый чат", callback_data="new_chat"),
            InlineKeyboardButton(text="👤 Профиль", callback_data="profile")
        ],
        [
            InlineKeyboardButton(text="💎 Тарифы", callback_data="plans"),
            InlineKeyboardButton(text="🤖 Модель", callback_data="models")
        ]
    ]
)


models_menu = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="🧠 ChatGPT 5", callback_data="set_model_gpt")
        ],
        [
            InlineKeyboardButton(text="🟣 Claude", callback_data="set_model_claude")
        ],
        [
            InlineKeyboardButton(text="🔵 Gemini", callback_data="set_model_gemini")
        ],
        [
            InlineKeyboardButton(text="⚫ DeepSeek", callback_data="set_model_deepseek")
        ]
    ]
)


def get_week_start():
    today = date.today()
    return today - timedelta(days=today.weekday())


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


async def get_or_create_user_by_data(telegram_id, username=None, first_name=None):
    today = date.today()
    week_start = get_week_start()

    async with db_pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT * FROM users WHERE telegram_id=$1",
            telegram_id
        )

        if not user:
            await conn.execute("""
                INSERT INTO users 
                (telegram_id, username, first_name, day_start, week_start)
                VALUES ($1, $2, $3, $4, $5)
            """, telegram_id, username, first_name, today, week_start)

        user = await conn.fetchrow(
            "SELECT * FROM users WHERE telegram_id=$1",
            telegram_id
        )

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

        return await conn.fetchrow(
            "SELECT * FROM users WHERE telegram_id=$1",
            telegram_id
        )


async def get_or_create_user(message: Message):
    return await get_or_create_user_by_data(
        telegram_id=message.from_user.id,
        username=message.from_user.username,
        first_name=message.from_user.first_name
    )


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

    return [
        {"role": row["role"], "content": row["content"]}
        for row in reversed(rows)
    ]


async def user_profile_text(user):
    plan = user["plan"]

    model_names = {
        "gpt": "ChatGPT 5",
        "claude": "Claude",
        "gemini": "Gemini",
        "deepseek": "DeepSeek"
    }

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


@dp.message(CommandStart())
async def start_handler(message: Message):
    await get_or_create_user(message)

    await message.answer(
        "🚀 Добро пожаловать в ChatGPT + Claude + Nano Banana!\n\n"
        "Напишите любой вопрос, и я отвечу с помощью AI.",
        reply_markup=menu
    )


@dp.callback_query(F.data == "profile")
async def profile_callback(callback: CallbackQuery):
    user = await get_or_create_user_by_data(
        telegram_id=callback.from_user.id,
        username=callback.from_user.username,
        first_name=callback.from_user.first_name
    )

    await callback.message.answer(
        await user_profile_text(user),
        reply_markup=menu
    )

    await callback.answer()


@dp.callback_query(F.data == "plans")
async def plans_callback(callback: CallbackQuery):
    await callback.message.answer(
        "💎 Тарифы\n\n"
        "FREE — 105 сообщений в неделю\n"
        "PRO — 1000 сообщений в день\n"
        "VIP — безлимит\n\n"
        "Оплата Telegram Stars скоро появится.",
        reply_markup=menu
    )

    await callback.answer()


@dp.callback_query(F.data == "models")
async def models_callback(callback: CallbackQuery):
    await callback.message.answer(
        "🤖 Выберите модель:",
        reply_markup=models_menu
    )

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
        "deepseek": "⚫ DeepSeek"
    }

    await callback.message.answer(
        f"✅ Модель переключена:\n\n{names.get(model)}",
        reply_markup=menu
    )

    await callback.answer()


@dp.callback_query(F.data == "new_chat")
async def new_chat_callback(callback: CallbackQuery):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM messages WHERE telegram_id=$1",
            callback.from_user.id
        )

    await callback.message.answer(
        "💬 Новый чат начат.\n\n"
        "Контекст очищен.",
        reply_markup=menu
    )

    await callback.answer()


@dp.message(F.text)
async def chat_handler(message: Message):
    user = await get_or_create_user(message)

    allowed, reason = await check_limit(user)

    if not allowed:
        if reason == "FREE_DAY_LIMIT":
            text = (
                "⏳ Дневной лимит FREE на сегодня закончился.\n\n"
                "Завтра сообщения обновятся автоматически."
            )
        elif reason == "FREE_WEEK_LIMIT":
            text = (
                "⏳ Недельный лимит FREE закончился.\n\n"
                "Лимит обновится на следующей неделе."
            )
        else:
            text = (
                "⏳ Дневной лимит PRO на сегодня закончился."
            )

        await message.answer(text, reply_markup=menu)
        return

    wait_message = await message.answer("Печатает ответ...")

    try:
        await save_message(message.from_user.id, "user", message.text)

        history = await get_chat_history(message.from_user.id)

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Ты полезный AI-ассистент. Отвечай профессионально и понятно."
                },
                *history
            ]
        )

        answer = response.choices[0].message.content

        await save_message(message.from_user.id, "assistant", answer)

        await increase_usage(message.from_user.id)

        await wait_message.edit_text(answer)

    except Exception as e:
        print(f"AI ERROR: {e}")

        await wait_message.edit_text(
            "⚠️ Сейчас у AI временные технические работы.\n\n"
            "Попробуйте ещё раз через минуту."
        )


async def main():
    print("BOT STARTING")

    await init_db()

    print("DATABASE CONNECTED")
    print("BOT STARTED")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
