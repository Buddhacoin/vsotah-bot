import asyncio
import base64
import os
from datetime import datetime
from io import BytesIO

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5.4-mini")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.4-mini")
ANTHROPIC_TEXT_MODEL = os.getenv("ANTHROPIC_TEXT_MODEL", "claude-sonnet-4-6")
ANTHROPIC_VISION_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-6")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
NANO_BANANA_MODEL = os.getenv("NANO_BANANA_MODEL", "imagen-4.0-generate-001")
GPT_IMAGE_MODEL = os.getenv("GPT_IMAGE_MODEL", "gpt-image-1")
GPT_IMAGE_QUALITY = os.getenv("GPT_IMAGE_QUALITY", "high")

TEXT_HISTORY_LIMIT = int(os.getenv("TEXT_HISTORY_LIMIT", "6"))
VISION_HISTORY_LIMIT = int(os.getenv("VISION_HISTORY_LIMIT", "2"))
TEXT_MAX_TOKENS = int(os.getenv("TEXT_MAX_TOKENS", "1200"))
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "900"))
AI_TIMEOUT_SECONDS = int(os.getenv("AI_TIMEOUT_SECONDS", "75"))
FILE_TEXT_LIMIT = int(os.getenv("FILE_TEXT_LIMIT", "18000"))

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
google_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
) if DEEPSEEK_API_KEY else None


def short_error_text(error: Exception) -> str:
    return str(error).replace("\n", " ")[:1200]


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


def system_prompt() -> str:
    current_datetime = datetime.now().strftime("%d.%m.%Y %H:%M")
    today_text = datetime.now().strftime("%A, %d.%m.%Y")
    return (
        "Ты профессиональный AI-ассистент. "
        "Отвечай понятно, структурно и по делу. "
        "Если пользователь пишет на русском — отвечай на русском. "
        "Если пользователь отправил изображение, внимательно проанализируй его и ответь на вопрос. "
        f"Текущая дата и время: {current_datetime}. Сегодня: {today_text}."
    )


async def ai_router(selected_model: str, messages: list[dict]):
    system_text = system_prompt()

    if selected_model == "claude":
        if not anthropic_client:
            return "⚠️ Claude пока не подключён. Администратору нужно добавить ANTHROPIC_API_KEY в Railway."

        response = await anthropic_client.messages.create(
            model=ANTHROPIC_TEXT_MODEL,
            max_tokens=TEXT_MAX_TOKENS,
            temperature=0.5,
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
            temperature=0.5,
        )
        return response.choices[0].message.content

    full_messages = [{"role": "system", "content": system_text}, *messages]
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            messages=full_messages,
            temperature=0.5,
            max_completion_tokens=TEXT_MAX_TOKENS,
        ),
        timeout=AI_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content


async def vision_router(selected_model: str, question: str, image_bytes: bytes, history: list[dict]):
    system_text = system_prompt()
    question = question.strip() or "Что изображено на фото? Опиши подробно и помоги пользователю."
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    if selected_model == "claude":
        if not anthropic_client:
            return "⚠️ Claude Vision пока не подключён. Администратору нужно добавить ANTHROPIC_API_KEY в Railway."

        response = await anthropic_client.messages.create(
            model=ANTHROPIC_VISION_MODEL,
            max_tokens=VISION_MAX_TOKENS,
            temperature=0.5,
            system=system_text,
            messages=[
                *normalize_anthropic_messages(history[-VISION_HISTORY_LIMIT:]),
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": question},
                    ],
                },
            ],
        )
        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()

    if selected_model == "gemini":
        if not google_client:
            return "⚠️ Gemini Vision пока не подключён. Администратору нужно добавить GOOGLE_API_KEY в Railway."

        prompt = (
            f"{system_text}\n\n"
            f"Краткая история диалога:\n{messages_to_plain_text(history[-VISION_HISTORY_LIMIT:])}\n\n"
            f"Вопрос пользователя к изображению: {question}"
        )

        def run_gemini_vision():
            response = google_client.models.generate_content(
                model=GEMINI_VISION_MODEL,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
            )
            return getattr(response, "text", "") or ""

        return (await asyncio.to_thread(run_gemini_vision)).strip()

    if selected_model in {"nanobanana", "gptimage"}:
        return (
            "📷 Вы сейчас выбрали генерацию изображений.\n\n"
            "Чтобы задать вопрос по фото, выберите ChatGPT, Claude или Gemini в меню «Выбрать нейросеть»."
        )

    openai_messages = [{"role": "system", "content": system_text}]
    for msg in history[-VISION_HISTORY_LIMIT:]:
        if msg.get("role") in {"user", "assistant"} and msg.get("content"):
            openai_messages.append({"role": msg["role"], "content": msg["content"]})

    openai_messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "low",
                    },
                },
            ],
        }
    )

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=openai_messages,
            temperature=0.4,
            max_completion_tokens=VISION_MAX_TOKENS,
        ),
        timeout=AI_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content


async def enhance_image_prompt(user_prompt: str, image_model: str = "image") -> str:
    """Translate and strengthen image prompts before sending them to image models."""
    original = (user_prompt or "").strip()
    if not original:
        original = "Create a high quality detailed image."

    system = (
        "You are a professional prompt engineer for AI image generation. "
        "Translate the user's request to clear English and make it specific for image generation. "
        "Preserve the user's main subject EXACTLY. If the user asks for an elephant, the image MUST contain an elephant, not a cat or another animal. "
        "Do not add random text, signs, logos, watermarks, captions, or fake letters unless the user explicitly asks for text. "
        "Keep the prompt concise but vivid: subject, scene, composition, lighting, style, quality. "
        "Return ONLY the final English image prompt, without explanations."
    )

    user = (
        f"Image model: {image_model}\n"
        f"User request: {original}\n\n"
        "Make a reliable English prompt. The generated image must follow the user request literally."
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=OPENAI_TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=500,
            ),
            timeout=15,
        )
        improved = (response.choices[0].message.content or "").strip()
        if improved:
            return improved[:2500]
    except Exception as e:
        print(f"IMAGE PROMPT ENHANCE ERROR: {short_error_text(e)}")

    return (
        f"Create a high quality, detailed image that follows this request exactly: {original}. "
        "Do not add random text, captions, watermarks, logos, or fake letters unless explicitly requested."
    )[:2500]


async def enhance_image_edit_prompt(user_prompt: str) -> str:
    """Translate and strengthen image edit prompts before sending them to image editing."""
    original = (user_prompt or "").strip() or "Improve this image while keeping the same subject."

    system = (
        "You are a professional prompt engineer for AI image editing. "
        "Translate the user's request to clear English. "
        "The edit must preserve the original subject, face, identity, pose, and important details unless the user explicitly asks to change them. "
        "Do not add random text, signs, watermarks, logos, or fake letters. "
        "Return ONLY the final English edit instruction, without explanations."
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=OPENAI_TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": original},
                ],
                max_completion_tokens=400,
            ),
            timeout=15,
        )
        improved = (response.choices[0].message.content or "").strip()
        if improved:
            return improved[:2000]
    except Exception as e:
        print(f"IMAGE EDIT PROMPT ENHANCE ERROR: {short_error_text(e)}")

    return (
        f"Edit this image according to the request: {original}. "
        "Preserve the original subject and identity. Do not add random text or watermarks."
    )[:2000]


async def generate_nano_banana_image(prompt: str) -> tuple[bytes | None, str]:
    if not google_client:
        return None, "⚠️ Nano Banana пока не подключён. Администратору нужно добавить GOOGLE_API_KEY в Railway."

    enhanced_prompt = await enhance_image_prompt(prompt, "Nano Banana / Gemini Image")

    def run_image_generation():
        response = google_client.models.generate_images(
            model=NANO_BANANA_MODEL,
            prompt=enhanced_prompt,
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

    enhanced_prompt = await enhance_image_prompt(prompt, "Sora GPT Image / OpenAI Image")

    response = await client.images.generate(
        model=GPT_IMAGE_MODEL,
        prompt=enhanced_prompt,
        size="1024x1024",
        quality=GPT_IMAGE_QUALITY,
        n=1,
    )

    item = response.data[0]
    if getattr(item, "b64_json", None):
        return normalize_b64(item.b64_json), "🌀 Готово"

    return None, "⚠️ Sora GPT Image не вернул изображение. Попробуйте другой запрос."


async def edit_gpt_image(prompt: str, image_bytes: bytes) -> tuple[bytes | None, str]:
    def normalize_b64(value: str) -> bytes:
        if value.startswith("data:image"):
            value = value.split(",", 1)[1]
        return base64.b64decode(value)

    enhanced_prompt = await enhance_image_edit_prompt(prompt)

    image_file = BytesIO(image_bytes)
    image_file.name = "input.png"

    # В текущей версии OpenAI Images API метод edit не принимает quality.
    # quality оставляем только для generate, иначе будет ошибка:
    # AsyncImages.edit() got an unexpected keyword argument 'quality'
    response = await client.images.edit(
        model=GPT_IMAGE_MODEL,
        image=image_file,
        prompt=enhanced_prompt,
        size="1024x1024",
        n=1,
    )

    item = response.data[0]
    if getattr(item, "b64_json", None):
        return normalize_b64(item.b64_json), "🖼 Готово"

    return None, "⚠️ Редактирование изображения не вернуло результат. Попробуйте другой запрос."


def render_pdf_pages_to_images(file_bytes: bytes, max_pages: int = 3) -> list[bytes]:
    """Рендерит первые страницы PDF в JPG для анализа сканов/чеков через Vision."""
    try:
        import fitz
    except Exception:
        return []

    images: list[bytes] = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_index in range(min(len(doc), max_pages)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            images.append(pix.tobytes("jpeg"))
        doc.close()
    except Exception as e:
        print(f"PDF RENDER ERROR: {e}")
        return []

    return images


async def analyze_pdf_images_with_openai(question: str, filename: str, file_bytes: bytes, history: list[dict]) -> str:
    """Fallback для PDF без извлекаемого текста: сканы, чеки, регистрации, картинки внутри PDF."""
    page_images = await asyncio.to_thread(render_pdf_pages_to_images, file_bytes, 3)
    if not page_images:
        return ""

    question = question.strip() or "Проанализируй PDF, прочитай видимый текст и выдели главное."
    content = [
        {
            "type": "text",
            "text": (
                f"Пользователь отправил PDF-файл: {filename}.\n"
                f"PDF не содержит обычного текстового слоя или плохо читается как текст, поэтому ниже страницы как изображения.\n\n"
                f"Задача пользователя: {question}\n\n"
                "Прочитай всё, что видно на страницах, и ответь по делу."
            ),
        }
    ]

    for img in page_images:
        image_b64 = base64.b64encode(img).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "low",
                },
            }
        )

    openai_messages = [{"role": "system", "content": system_prompt()}]
    for msg in history[-VISION_HISTORY_LIMIT:]:
        if msg.get("role") in {"user", "assistant"} and msg.get("content"):
            openai_messages.append({"role": msg["role"], "content": msg["content"]})
    openai_messages.append({"role": "user", "content": content})

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=openai_messages,
            temperature=0.4,
            max_completion_tokens=VISION_MAX_TOKENS,
        ),
        timeout=AI_TIMEOUT_SECONDS,
    )
    return response.choices[0].message.content or ""


async def file_router(selected_model: str, question: str, filename: str, extracted_text: str, history: list[dict]):
    question = question.strip() or "Проанализируй файл, сделай краткое резюме и выдели главное."
    content = (
        f"Пользователь отправил файл: {filename}\n\n"
        f"Вопрос пользователя: {question}\n\n"
        f"Текст/данные из файла:\n{extracted_text}"
    )
    messages = [*history[-TEXT_HISTORY_LIMIT:], {"role": "user", "content": content}]
    return await ai_router(selected_model, messages)

