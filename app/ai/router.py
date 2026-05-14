import asyncio
import base64
import os
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Callable, Coroutine

import httpx

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types

from app.ai.prompts import system_prompt, clean_ai_answer
from app.ai.memory import build_memory
from app.ai.personalities import get_personality
from app.ai.image_router import (
    build_image_generation_prompt,
    build_image_edit_prompt,
    image_result_caption,
    infer_gemini_aspect_ratio,
    infer_openai_image_size,
)
from app.ai.vision_router import build_vision_prompt, build_pdf_vision_prompt


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Модели можно менять через Railway Variables без правки кода.
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
ANTHROPIC_TEXT_MODEL = os.getenv("ANTHROPIC_TEXT_MODEL", "claude-3-5-sonnet-latest")
ANTHROPIC_VISION_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", "claude-3-5-sonnet-latest")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
DEEPSEEK_TEXT_MODEL = os.getenv("DEEPSEEK_TEXT_MODEL", "deepseek-chat")
NANO_BANANA_MODEL = os.getenv("NANO_BANANA_MODEL", "imagen-4.0-generate-001")
GPT_IMAGE_MODEL = os.getenv("GPT_IMAGE_MODEL", "gpt-image-1")
GPT_IMAGE_QUALITY = os.getenv("GPT_IMAGE_QUALITY", "high")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LIVE_WEB_TIMEOUT = float(os.getenv("LIVE_WEB_TIMEOUT", "12"))

TEXT_HISTORY_LIMIT = int(os.getenv("TEXT_HISTORY_LIMIT", "6"))
VISION_HISTORY_LIMIT = int(os.getenv("VISION_HISTORY_LIMIT", "2"))
TEXT_MAX_TOKENS = int(os.getenv("TEXT_MAX_TOKENS", "1200"))
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "900"))
AI_TIMEOUT_SECONDS = int(os.getenv("AI_TIMEOUT_SECONDS", "75"))
FILE_TEXT_LIMIT = int(os.getenv("FILE_TEXT_LIMIT", "18000"))

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
google_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

deepseek_client = (
    AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    if DEEPSEEK_API_KEY
    else None
)


# ---------------- AI CORE 3.0: live-data + quality layer ----------------
LIVE_WEB_KEYWORDS = (
    "сейчас", "сегодня", "последн", "актуальн", "новост", "курс", "цена", "стоимость",
    "погода", "прогноз", "расписание", "результат", "кто выиграл", "где находится",
    "куда летал", "что произошло", "что случилось", "2025", "2026",
)

WEATHER_KEYWORDS = ("погода", "прогноз", "температура", "осадки", "ветер")
CRYPTO_KEYWORDS = ("биткоин", "bitcoin", "btc", "эфир", "ethereum", "eth", "крипт", "coin")


def latest_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == "user" and msg.get("content"):
            return str(msg["content"]).strip()
    return ""


def needs_live_web(question: str) -> bool:
    q = (question or "").lower()
    if not q:
        return False
    if any(k in q for k in LIVE_WEB_KEYWORDS):
        return True
    # Даты/относительные формулировки часто требуют свежих данных.
    if re.search(r"\b(20\d{2}|май|июн|июл|август|сентябр|октябр|ноябр|декабр|январ|феврал|март|апрел)\b", q):
        return True
    return False


def is_weather_query(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in WEATHER_KEYWORDS)


def is_crypto_query(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in CRYPTO_KEYWORDS)


def guess_weather_location(question: str) -> str:
    text = question or ""
    # Берём всё после популярных предлогов, но без лишней пунктуации.
    patterns = [
        r"погод[ауы]?\s+(?:сейчас\s+)?(?:в|во|на)\s+(.+)",
        r"температур[аы]?\s+(?:сейчас\s+)?(?:в|во|на)\s+(.+)",
        r"прогноз\s+(?:погоды\s+)?(?:в|во|на)\s+(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            loc = m.group(1)
            loc = re.sub(r"[?.!,]+$", "", loc).strip()
            loc = re.sub(r"\s+", " ", loc)
            return loc[:120]
    return text.strip()[:120]


async def fetch_json(url: str, params: dict | None = None, headers: dict | None = None, timeout: float = LIVE_WEB_TIMEOUT) -> dict:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


async def get_live_weather_answer(question: str) -> str:
    location = guess_weather_location(question)
    if not location:
        return ""
    try:
        geo = await fetch_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "ru", "format": "json"},
        )
        results = geo.get("results") or []
        if not results:
            return ""
        place = results[0]
        lat, lon = place.get("latitude"), place.get("longitude")
        name = place.get("name") or location
        admin1 = place.get("admin1") or ""
        country = place.get("country") or ""
        weather = await fetch_json(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,relative_humidity_2m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 1,
            },
        )
        cur = weather.get("current") or {}
        daily = (weather.get("daily") or {})
        max_t = (daily.get("temperature_2m_max") or [None])[0]
        min_t = (daily.get("temperature_2m_min") or [None])[0]
        rain = (daily.get("precipitation_probability_max") or [None])[0]
        parts = []
        place_line = ", ".join([x for x in [name, admin1, country] if x])
        parts.append(f"Сейчас в {place_line}:")
        if cur.get("temperature_2m") is not None:
            parts.append(f"температура {round(cur['temperature_2m'])}°C")
        if cur.get("apparent_temperature") is not None:
            parts.append(f"ощущается как {round(cur['apparent_temperature'])}°C")
        if cur.get("wind_speed_10m") is not None:
            parts.append(f"ветер {round(cur['wind_speed_10m'])} км/ч")
        if cur.get("relative_humidity_2m") is not None:
            parts.append(f"влажность {round(cur['relative_humidity_2m'])}%")
        answer = "; ".join(parts) + "."
        if max_t is not None or min_t is not None or rain is not None:
            answer += "\n\nНа сегодня: "
            day_bits = []
            if max_t is not None:
                day_bits.append(f"днём до {round(max_t)}°C")
            if min_t is not None:
                day_bits.append(f"минимум около {round(min_t)}°C")
            if rain is not None:
                day_bits.append(f"вероятность осадков до {round(rain)}%")
            answer += "; ".join(day_bits) + "."
        return answer
    except Exception as e:
        print(f"LIVE WEATHER ERROR: {short_error_text(e)}")
        return ""


async def get_crypto_answer(question: str) -> str:
    q = (question or "").lower()
    coin_id = "bitcoin"
    coin_name = "Bitcoin"
    if "эфир" in q or "ethereum" in q or re.search(r"\beth\b", q):
        coin_id = "ethereum"
        coin_name = "Ethereum"
    try:
        data = await fetch_json(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": coin_id, "vs_currencies": "usd,rub,eur", "include_24hr_change": "true"},
        )
        item = data.get(coin_id) or {}
        usd = item.get("usd")
        rub = item.get("rub")
        eur = item.get("eur")
        change = item.get("usd_24h_change")
        if usd is None:
            return ""
        def money(v, symbol):
            if v is None:
                return None
            return f"{symbol}{v:,.0f}" if symbol != "₽" else f"{v:,.0f} ₽".replace(",", " ")
        lines = [f"Сейчас {coin_name}: {money(usd, '$')} / {money(rub, '₽')} / {money(eur, '€')}."]
        if change is not None:
            sign = "+" if change >= 0 else ""
            lines.append(f"Изменение за 24 часа: {sign}{change:.2f}%.")
        lines.append("Цена быстро меняется, источник: CoinGecko.")
        return "\n".join(lines)
    except Exception as e:
        print(f"CRYPTO LIVE ERROR: {short_error_text(e)}")
        return ""


def build_search_query(question: str) -> str:
    q = (question or "").strip()
    q = re.sub(r"\s+", " ", q)
    # Подсказываем поиску, что нужен русский ответ, но не ломаем исходный вопрос.
    if re.search(r"[а-яА-ЯёЁ]", q):
        return f"{q} актуально сегодня"
    return q


async def tavily_search(question: str) -> list[dict]:
    if not TAVILY_API_KEY:
        return []
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": build_search_query(question),
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
        "max_results": 5,
    }
    try:
        async with httpx.AsyncClient(timeout=LIVE_WEB_TIMEOUT) as client:
            response = await client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()
        results = []
        for item in data.get("results") or []:
            title = _safe_text(item.get("title"))[:160]
            url = _safe_text(item.get("url"))[:300]
            content = _safe_text(item.get("content"))[:900]
            if title and content:
                results.append({"title": title, "url": url, "content": content})
        return results
    except Exception as e:
        print(f"TAVILY SEARCH ERROR: {short_error_text(e)}")
        return []


def format_web_context(results: list[dict]) -> str:
    if not results:
        return ""
    lines = []
    for i, item in enumerate(results[:5], start=1):
        lines.append(
            f"Источник {i}: {item.get('title')}\n"
            f"URL: {item.get('url')}\n"
            f"Фрагмент: {item.get('content')}"
        )
    return "\n\n".join(lines)


def build_quality_system_prompt(selected_model: str, web_context: str = "") -> str:
    base = system_prompt(get_personality(selected_model))
    quality = (
        "\n\nAI CORE 3.0 правила качества:\n"
        "1. Отвечай на русском, если пользователь пишет по-русски.\n"
        "2. Не выдумывай факты. Если свежих или надёжных данных нет — честно скажи, что не удалось надёжно проверить.\n"
        "3. Не упоминай API, ключи, Tavily, Railway, админа, системные настройки.\n"
        "4. Не вставляй длинные списки ссылок. Если источники нужны — максимум 2 коротких строки в конце.\n"
        "5. Сначала дай прямой ответ, потом краткое пояснение.\n"
        "6. Не пиши 'если нужно, дай знать' в каждом ответе.\n"
        "7. Для актуальных вопросов опирайся только на предоставленный web-context.\n"
    )
    if web_context:
        quality += (
            "\nWeb-context для ответа ниже. Используй его как источники, но не копируй сырой текст. "
            "Если источники не отвечают на вопрос, скажи, что не удалось надёжно проверить.\n\n"
            f"{web_context}"
        )
    return base + quality


class ProviderUnavailable(Exception):
    pass


def short_error_text(error: Exception) -> str:
    return str(error).replace("\n", " ")[:1200]


def _safe_text(value: Any) -> str:
    return (value or "").strip()


def _extract_anthropic_text(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "\n".join(part for part in parts if part).strip()


def normalize_anthropic_messages(messages: list[dict]) -> list[dict]:
    result: list[dict] = []
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


def normalize_openai_messages(messages: list[dict]) -> list[dict]:
    result: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in {"system", "user", "assistant"} and content:
            result.append({"role": role, "content": content})
    return result


def messages_to_plain_text(messages: list[dict]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def provider_order(selected_model: str, task: str = "text") -> list[str]:
    """
    AI Router 3.0: основной провайдер зависит от выбранной модели,
    дальше идут fallback-провайдеры, если API временно недоступен.
    """
    selected_model = (selected_model or "gpt").lower()

    if task == "vision":
        orders = {
            "gpt": ["openai", "gemini", "anthropic"],
            "claude": ["anthropic", "openai", "gemini"],
            "gemini": ["gemini", "openai", "anthropic"],
            "deepseek": ["openai", "gemini", "anthropic"],
        }
        return orders.get(selected_model, ["openai", "gemini", "anthropic"])

    orders = {
        "gpt": ["openai", "gemini", "anthropic", "deepseek"],
        "claude": ["anthropic", "openai", "gemini", "deepseek"],
        "gemini": ["gemini", "openai", "anthropic", "deepseek"],
        "deepseek": ["deepseek", "openai", "gemini", "anthropic"],
    }
    return orders.get(selected_model, ["openai", "gemini", "anthropic", "deepseek"])


async def run_with_fallback(
    providers: list[str],
    calls: dict[str, Callable[[], Coroutine[Any, Any, str]]],
    task_name: str,
) -> str:
    errors: list[str] = []

    for provider in providers:
        call = calls.get(provider)
        if not call:
            continue
        try:
            result = await asyncio.wait_for(call(), timeout=AI_TIMEOUT_SECONDS)
            result = clean_ai_answer(_safe_text(result))
            if result:
                if errors:
                    print(f"AI ROUTER FALLBACK OK | task={task_name} | provider={provider} | previous={'; '.join(errors)}")
                return result
            errors.append(f"{provider}: empty response")
        except ProviderUnavailable as e:
            errors.append(f"{provider}: {short_error_text(e)}")
        except Exception as e:
            errors.append(f"{provider}: {short_error_text(e)}")
            print(f"AI ROUTER PROVIDER ERROR | task={task_name} | provider={provider} | {short_error_text(e)}")

    print(f"AI ROUTER ALL FAILED | task={task_name} | {'; '.join(errors)}")
    return (
        "⚠️ Сейчас нейросети временно недоступны.\n\n"
        "Попробуйте ещё раз чуть позже или выберите другую нейросеть в меню."
    )


async def openai_chat(messages: list[dict], model: str, max_tokens: int, temperature: float = 0.45) -> str:
    if not openai_client:
        raise ProviderUnavailable("OPENAI_API_KEY не задан")

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
    except TypeError:
        # Совместимость со старыми/другими OpenAI-compatible моделями.
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return _safe_text(response.choices[0].message.content)


async def deepseek_chat(messages: list[dict], max_tokens: int, temperature: float = 0.45) -> str:
    if not deepseek_client:
        raise ProviderUnavailable("DEEPSEEK_API_KEY не задан")

    response = await deepseek_client.chat.completions.create(
        model=DEEPSEEK_TEXT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _safe_text(response.choices[0].message.content)


async def anthropic_chat(system_text: str, messages: list[dict], model: str, max_tokens: int, temperature: float = 0.45) -> str:
    if not anthropic_client:
        raise ProviderUnavailable("ANTHROPIC_API_KEY не задан")

    response = await anthropic_client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_text,
        messages=normalize_anthropic_messages(messages),
    )
    return _extract_anthropic_text(response)


async def gemini_chat(system_text: str, messages: list[dict], model: str) -> str:
    if not google_client:
        raise ProviderUnavailable("GOOGLE_API_KEY не задан")

    prompt = f"{system_text}\n\nКраткая память диалога:\n{messages_to_plain_text(messages)}"

    def run_gemini() -> str:
        response = google_client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return getattr(response, "text", "") or ""

    return await asyncio.to_thread(run_gemini)


async def ai_router(selected_model: str, messages: list[dict]) -> str:
    memory = build_memory(messages, limit=TEXT_HISTORY_LIMIT)
    question = latest_user_text(memory)

    # Fast reliable live-data paths for common current-data tasks.
    if is_weather_query(question):
        weather_answer = await get_live_weather_answer(question)
        if weather_answer:
            return clean_ai_answer(weather_answer)

    if is_crypto_query(question):
        crypto_answer = await get_crypto_answer(question)
        if crypto_answer:
            return clean_ai_answer(crypto_answer)

    web_context = ""
    if needs_live_web(question):
        results = await tavily_search(question)
        web_context = format_web_context(results)

    system_text = build_quality_system_prompt(selected_model, web_context)
    openai_messages = normalize_openai_messages([{"role": "system", "content": system_text}, *memory])

    calls = {
        "openai": lambda: openai_chat(openai_messages, OPENAI_TEXT_MODEL, TEXT_MAX_TOKENS, 0.35),
        "anthropic": lambda: anthropic_chat(system_text, memory, ANTHROPIC_TEXT_MODEL, TEXT_MAX_TOKENS, 0.35),
        "gemini": lambda: gemini_chat(system_text, memory, GEMINI_TEXT_MODEL),
        "deepseek": lambda: deepseek_chat(openai_messages, TEXT_MAX_TOKENS, 0.35),
    }
    return await run_with_fallback(provider_order(selected_model, "text"), calls, f"text:{selected_model}")


async def openai_vision(system_text: str, question: str, image_bytes: bytes, history: list[dict]) -> str:
    if not openai_client:
        raise ProviderUnavailable("OPENAI_API_KEY не задан")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    openai_messages: list[dict] = [{"role": "system", "content": system_text}]

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

    return await openai_chat(openai_messages, OPENAI_VISION_MODEL, VISION_MAX_TOKENS, 0.4)


async def anthropic_vision(system_text: str, question: str, image_bytes: bytes, history: list[dict]) -> str:
    if not anthropic_client:
        raise ProviderUnavailable("ANTHROPIC_API_KEY не задан")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
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
    return _extract_anthropic_text(response)


async def gemini_vision(system_text: str, question: str, image_bytes: bytes, history: list[dict]) -> str:
    if not google_client:
        raise ProviderUnavailable("GOOGLE_API_KEY не задан")

    prompt = (
        f"{system_text}\n\n"
        f"Краткая история диалога:\n{messages_to_plain_text(history[-VISION_HISTORY_LIMIT:])}\n\n"
        f"Вопрос пользователя к изображению: {question}"
    )

    def run_gemini_vision() -> str:
        response = google_client.models.generate_content(
            model=GEMINI_VISION_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ],
        )
        return getattr(response, "text", "") or ""

    return await asyncio.to_thread(run_gemini_vision)


async def vision_router(selected_model: str, question: str, image_bytes: bytes, history: list[dict]) -> str:
    if selected_model in {"nanobanana", "gptimage"}:
        return (
            "📷 Вы сейчас выбрали генерацию изображений.\n\n"
            "Чтобы задать вопрос по фото, выберите ChatGPT, Claude или Gemini в меню «Выбрать AI»."
        )

    system_text = system_prompt(get_personality(selected_model))
    question = build_vision_prompt(question)

    calls = {
        "openai": lambda: openai_vision(system_text, question, image_bytes, history),
        "anthropic": lambda: anthropic_vision(system_text, question, image_bytes, history),
        "gemini": lambda: gemini_vision(system_text, question, image_bytes, history),
    }
    return await run_with_fallback(provider_order(selected_model, "vision"), calls, f"vision:{selected_model}")


async def enhance_image_prompt(user_prompt: str, image_model: str = "image") -> str:
    """Image/Vision Pipeline 2.1: smart deterministic prompt + optional AI polishing."""
    original = (user_prompt or "").strip() or "Create a high quality detailed image."
    deterministic_prompt = build_image_generation_prompt(original, image_model)

    system = (
        "You are a professional prompt engineer for AI image generation. "
        "Translate the user's request to clear English and make it specific for image generation. "
        "Preserve the user's main subject EXACTLY. If the user asks for an elephant, the image MUST contain an elephant, not a cat or another animal. "
        "Do not add random text, signs, logos, watermarks, captions, or fake letters unless the user explicitly asks for text. "
        "Respect the target aspect ratio and image type from the provided plan. "
        "Keep the prompt concise but vivid: subject, scene, composition, lighting, style, quality. "
        "Return ONLY the final English image prompt, without explanations."
    )

    user = (
        f"Image model: {image_model}\n"
        f"User request: {original}\n\n"
        f"VSotahBot image plan:\n{deterministic_prompt}\n\n"
        "Make a reliable English prompt. The generated image must follow the user request literally."
    )

    try:
        if not openai_client:
            raise ProviderUnavailable("OPENAI_API_KEY не задан")
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
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

    return deterministic_prompt[:2500]


async def enhance_image_edit_prompt(user_prompt: str) -> str:
    """Image/Vision Pipeline 2.1: smart edit instructions + optional AI polishing."""
    original = (user_prompt or "").strip() or "Improve this image while keeping the same subject."
    deterministic_prompt = build_image_edit_prompt(original)

    system = (
        "You are a professional prompt engineer for AI image editing. "
        "Translate the user's request to clear English. "
        "The edit must preserve the original subject, face, identity, pose, and important details unless the user explicitly asks to change them. "
        "Do not add random text, signs, watermarks, logos, or fake letters. "
        "Return ONLY the final English edit instruction, without explanations."
    )

    try:
        if not openai_client:
            raise ProviderUnavailable("OPENAI_API_KEY не задан")
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model=OPENAI_TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": deterministic_prompt},
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

    return deterministic_prompt[:2000]


async def generate_nano_banana_image(prompt: str) -> tuple[bytes | None, str]:
    if not google_client:
        return None, "⚠️ Генерация временно недоступна. Попробуйте ещё раз чуть позже."

    enhanced_prompt = await enhance_image_prompt(prompt, "Nano Banana / Gemini Image")

    def run_image_generation() -> tuple[bytes | None, str]:
        response = google_client.models.generate_images(
            model=NANO_BANANA_MODEL,
            prompt=enhanced_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=infer_gemini_aspect_ratio(prompt),
                person_generation="allow_adult",
            ),
        )

        if not response.generated_images:
            return None, "⚠️ Генерация временно недоступна. Попробуйте ещё раз чуть позже."

        image_obj = response.generated_images[0].image

        if hasattr(image_obj, "image_bytes") and image_obj.image_bytes:
            return image_obj.image_bytes, image_result_caption("nanobanana")

        buffer = BytesIO()
        image_obj.save(buffer, format="PNG")
        return buffer.getvalue(), image_result_caption("nanobanana")

    try:
        return await asyncio.to_thread(run_image_generation)
    except Exception as e:
        print(f"NANO BANANA IMAGE ERROR: {short_error_text(e)}")
        return None, "⚠️ Генерация временно недоступна. Попробуйте ещё раз чуть позже."


async def generate_gpt_image(prompt: str) -> tuple[bytes | None, str]:
    if not openai_client:
        return None, "⚠️ Sora GPT Image пока не подключён. Администратору нужно добавить OPENAI_API_KEY в Railway."

    def normalize_b64(value: str) -> bytes:
        if value.startswith("data:image"):
            value = value.split(",", 1)[1]
        return base64.b64decode(value)

    enhanced_prompt = await enhance_image_prompt(prompt, "Sora GPT Image / OpenAI Image")

    response = await openai_client.images.generate(
        model=GPT_IMAGE_MODEL,
        prompt=enhanced_prompt,
        size=infer_openai_image_size(prompt),
        quality=GPT_IMAGE_QUALITY,
        n=1,
    )

    item = response.data[0]
    if getattr(item, "b64_json", None):
        return normalize_b64(item.b64_json), image_result_caption("gptimage")

    return None, "⚠️ Sora GPT Image не вернул изображение. Попробуйте другой запрос."


async def edit_gpt_image(prompt: str, image_bytes: bytes) -> tuple[bytes | None, str]:
    if not openai_client:
        return None, "⚠️ Редактирование изображений пока не подключено. Администратору нужно добавить OPENAI_API_KEY в Railway."

    def normalize_b64(value: str) -> bytes:
        if value.startswith("data:image"):
            value = value.split(",", 1)[1]
        return base64.b64decode(value)

    enhanced_prompt = await enhance_image_edit_prompt(prompt)

    image_file = BytesIO(image_bytes)
    image_file.name = "input.png"

    # В текущей версии OpenAI Images API метод edit не принимает quality.
    response = await openai_client.images.edit(
        model=GPT_IMAGE_MODEL,
        image=image_file,
        prompt=enhanced_prompt,
        size="1024x1024",
        n=1,
    )

    item = response.data[0]
    if getattr(item, "b64_json", None):
        return normalize_b64(item.b64_json), image_result_caption("gptimage", "edit")

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


async def analyze_pdf_images_with_openai(
    question: str,
    filename: str,
    file_bytes: bytes,
    history: list[dict],
    selected_model: str = "gpt",
) -> str:
    """Fallback для PDF без извлекаемого текста: сканы, чеки, регистрации, картинки внутри PDF."""
    page_images = await asyncio.to_thread(render_pdf_pages_to_images, file_bytes, 3)
    if not page_images:
        return ""

    if not openai_client:
        return ""

    question = question.strip() or "Проанализируй PDF, прочитай видимый текст и выдели главное."
    content: list[dict] = [
        {
            "type": "text",
            "text": build_pdf_vision_prompt(filename, question),
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

    system_text = system_prompt(get_personality(selected_model))
    openai_messages: list[dict] = [{"role": "system", "content": system_text}]
    for msg in history[-VISION_HISTORY_LIMIT:]:
        if msg.get("role") in {"user", "assistant"} and msg.get("content"):
            openai_messages.append({"role": msg["role"], "content": msg["content"]})
    openai_messages.append({"role": "user", "content": content})

    try:
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model=OPENAI_VISION_MODEL,
                messages=openai_messages,
                temperature=0.4,
                max_completion_tokens=VISION_MAX_TOKENS,
            ),
            timeout=AI_TIMEOUT_SECONDS,
        )
        return clean_ai_answer(response.choices[0].message.content or "")
    except Exception as e:
        print(f"PDF VISION ERROR: {short_error_text(e)}")
        return ""


async def file_router(selected_model: str, question: str, filename: str, extracted_text: str, history: list[dict]) -> str:
    question = question.strip() or "Проанализируй файл, сделай краткое резюме и выдели главное."
    extracted_text = (extracted_text or "")[:FILE_TEXT_LIMIT]
    content = (
        f"Пользователь отправил файл: {filename}\n\n"
        f"Вопрос пользователя: {question}\n\n"
        f"Текст/данные из файла:\n{extracted_text}"
    )
    messages = [*history[-TEXT_HISTORY_LIMIT:], {"role": "user", "content": content}]
    return await ai_router(selected_model, messages)



