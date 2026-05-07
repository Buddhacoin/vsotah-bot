import asyncio
import os
from datetime import date, timedelta

import asyncpg
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, KeyboardButton, ReplyKeyboardMarkup
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


menu = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="💬 Новый чат"), KeyboardButton(text="👤 Профиль")],
        [KeyboardButton(text="💎 Тарифы"), KeyboardButton(text="🤖 Модель")]
    ],
    resize_keyboard=True
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


async def get_or_create_user(message: Message):
    telegram_id = message.from_user.id
    username = message.from_user.username
    first_name = message.from_user.first_name

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

        user = await conn.fetchrow(
            "SELECT * FROM users WHERE telegram_id=$1",
            telegram_id
        )

        return user


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

    history = []

    for row in reversed(rows):
        history.append({
            "role": row["role"],
            "content": row["content"]
        })

    return history


async def user_profile_text(user):
    plan = user["plan"]

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


@dp.message(Command("profile"))
@dp.message(F.text == "👤 Профиль")
async def profile_handler(message: Message):
    user = await get_or_create_user(message)
    await message.answer(await user_profile_text(user), reply_markup=menu)


@dp.message(F.text == "💎 Тарифы")
async def plans_handler(message: Message):
    await message.answer(
        "💎 Тарифы\n\n"
        "FREE — 105 сообщений в неделю\n"
        "PRO — 1000 сообщений в день\n"
        "VIP — безлимит\n\n"
        "Оплата Telegram Stars будет добавлена следующим шагом.",
        reply_markup=menu
    )


@dp.message(F.text == "🤖 Модель")
async def model_handler(message: Message):
    await message.answer(
        "🤖 Сейчас активна модель: GPT 5\n\n"
        "Скоро добавим выбор:\n"
        "• GPT\n"
        "• Claude\n"
        "• Gemini\n"
        "• DeepSeek",
        reply_markup=menu
    )


@dp.message(F.text == "💬 Новый чат")
async def new_chat_handler(message: Message):
    async with db_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM messages WHERE telegram_id=$1",
            message.from_user.id
        )

    await message.answer(
        "💬 Новый чат начат.\n\n"
        "Контекст очищен. Напишите новый вопрос.",
        reply_markup=menu
    )


@dp.message(F.text)
async def chat_handler(message: Message):
    user = await get_or_create_user(message)

    allowed, reason = await check_limit(user)

    if not allowed:
        if reason == "FREE_DAY_LIMIT":
            text = (
                "⏳ Дневной лимит FREE на сегодня закончился.\n\n"
                "Завтра сообщения обновятся автоматически.\n"
                "Для большего лимита можно перейти на PRO."
            )
        elif reason == "FREE_WEEK_LIMIT":
            text = (
                "⏳ Недельный лимит FREE закончился.\n\n"
                "Лимит обновится на следующей неделе.\n"
                "Для продолжения можно перейти на PRO."
            )
        else:
            text = (
                "⏳ Дневной лимит PRO на сегодня закончился.\n\n"
                "Завтра сообщения обновятся автоматически."
            )

        await message.answer(text, reply_markup=menu)
        return

    wait_message = await message.answer("Печатает ответ...")

    try:
        await save_message(message.from_user.id, "user", message.text)

        history = await get_chat_history(message.from_user.id, limit=12)

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Ты полезный AI-ассистент. Отвечай понятно, профессионально и на языке пользователя. Помни контекст текущего диалога."
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
            "⚠️ Сейчас у GPT временные технические работы.\n\n"
            "Попробуйте ещё раз через минуту или выберите другую модель."
        )


async def main():
    print("BOT STARTING")

    await init_db()

    print("DATABASE CONNECTED")
    print("BOT STARTED")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
