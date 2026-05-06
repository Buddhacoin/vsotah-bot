import asyncio
import os

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from openai import AsyncOpenAI

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

client = AsyncOpenAI(
    api_key=OPENAI_API_KEY
)

@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.answer(
        "🚀 GPTClaude AI Bot запущен!\n\nНапиши любой вопрос."
    )

@dp.message(F.text)
async def chat_handler(message: Message):

    user_text = message.text

    wait_message = await message.answer("💭 Думаю...")

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": user_text
                }
            ]
        )

        answer = response.choices[0].message.content

        await wait_message.edit_text(answer)

    except Exception as e:
        await wait_message.edit_text(
            f"❌ Ошибка:\n{e}"
        )

async def main():
    print("BOT STARTED")

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
