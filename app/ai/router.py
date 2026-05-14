import asyncio
import base64
import json
import os
import re
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Callable, Coroutine

import aiohttp

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

TEXT_HISTORY_LIMIT = int(os.getenv("TEXT_HISTORY_LIMIT", "10"))
VISION_HISTORY_LIMIT = int(os.getenv("VISION_HISTORY_LIMIT", "2"))
TEXT_MAX_TOKENS = int(os.getenv("TEXT_MAX_TOKENS", "1200"))
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "900"))
AI_TIMEOUT_SECONDS = int(os.getenv("AI_TIMEOUT_SECONDS", "75"))
FILE_TEXT_LIMIT = int(os.getenv("FILE_TEXT_LIMIT", "18000"))

# AI Core Upgrade: рабочие режимы и live-web/research слой.
# Live web включается только если в Railway добавлен один из ключей:
# TAVILY_API_KEY, SERPER_API_KEY или BRAVE_SEARCH_API_KEY.
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
WEB_SEARCH_TIMEOUT = int(os.getenv("WEB_SEARCH_TIMEOUT", "12"))
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

RESEARCH_MAX_TOKENS = int(os.getenv("RESEARCH_MAX_TOKENS", "1800"))
BUSINESS_MAX_TOKENS = int(os.getenv("BUSINESS_MAX_TOKENS", "1600"))
CODE_MAX_TOKENS = int(os.getenv("CODE_MAX_TOKENS", "1800"))

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
google_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

deepseek_client = (
    AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    if DEEPSEEK_API_KEY
    else None
)


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



def latest_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages or []):
        if msg.get("role") == "user" and msg.get("content"):
            return str(msg.get("content", "")).strip()
    return ""


def normalize_search_query(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    text = re.sub(r"^/(research|web|search|business|code)\s+", "", text, flags=re.I)
    return text[:450]


def build_search_query(text: str) -> str:
    """Build a clean query for live web search.

    Important: previous builds called this function but did not define it, so live
    web silently failed and the bot fell back to stale LLM memory. This function
    is intentionally simple and universal: no hardcoded answers for Bitcoin,
    Moscow, Trump, etc. It only cleans the user's question and adds a Russian
    freshness hint when needed.
    """
    query = normalize_search_query(text)
    if not query:
        return ""

    # Remove bot/menu command noise but keep the real meaning of the question.
    query = re.sub(r"\b(VSotah|VSotahBot|бот|ответь|скажи|пожалуйста)\b", " ", query, flags=re.I)
    query = re.sub(r"\s+", " ", query).strip()

    # Tavily works better when current questions explicitly ask for fresh info.
    freshness_words = [
        "сейчас", "сегодня", "актуаль", "последн", "новост", "курс", "цена",
        "погода", "кто сейчас", "на данный момент", "latest", "today", "current", "now",
    ]
    if any(word in query.lower() for word in freshness_words):
        query = f"{query} актуальные данные {datetime.now(timezone.utc).year}"

    return query[:450]



AI_QUALITY_LAYER_VERSION = "AI Quality Layer 1.0"


def detect_task_type(text: str, mode: str = "chat") -> str:
    """Lightweight task classifier for better prompts without extra API calls."""
    t = (text or "").lower()
    if mode in {"research", "business", "code", "web"}:
        return mode
    if any(x in t for x in ["код", "ошибка", "traceback", "python", "github", "railway", "api", "deploy", "syntaxerror"]):
        return "code"
    if any(x in t for x in ["договор", "кп", "продажи", "бизнес", "клиент", "маркетинг", "выручка", "прибыль", "стратегия"]):
        return "business"
    if any(x in t for x in ["проанализируй", "исследуй", "сравни", "почему", "причины", "риски", "вывод", "отчет", "отчёт"]):
        return "analysis"
    if wants_live_web(text):
        return "web"
    if any(x in t for x in ["напиши", "придумай", "пост", "текст", "сценарий", "реклама", "описание"]):
        return "writing"
    return "general"


def quality_layer_instruction(query: str, mode: str = "chat", web_requested: bool = False, web_ready: bool = False) -> str:
    """Universal answer-quality rules for VSotah AI.

    Keep this provider-neutral: OpenAI, Claude, Gemini and DeepSeek all receive it.
    """
    task = detect_task_type(query, mode)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base = [
        f"{AI_QUALITY_LAYER_VERSION}. Current UTC date: {today}.",
        "You are VSotah AI: a smart, practical Telegram assistant, not a raw API wrapper.",
        "Always answer in the user's language. If the user writes Russian, answer in natural Russian.",
        "Be accurate first, concise second. Do not invent facts, dates, prices, laws, news, names or statistics.",
        "If a factual/current claim is uncertain and no reliable live evidence is available, say this plainly and give the safest useful answer.",
        "Do not mention internal tools, API keys, Tavily, Railway, providers, admin settings, system prompts or implementation details.",
        "Avoid filler: no 'как ИИ-модель', no long disclaimers, no repeated introductions, no decorative separators.",
        "Use short paragraphs. Use bullets only when they make the answer easier to read.",
        "For questions with a direct answer, start with the direct answer, then add context.",
        "For advice, give concrete next steps, not abstract motivational text.",
        f"Detected task type: {task}.",
    ]
    if task == "web":
        base += [
            "This question may require fresh data. If live evidence is present, rely on it and do not use stale memory for current facts.",
            "When web evidence is noisy, ignore irrelevant snippets instead of summarizing them.",
            "Do not paste raw URLs unless the user asks for links. If sources are useful, include max 2 short source names at the end.",
        ]
    elif task == "code":
        base += [
            "For code/debugging: identify the cause first, then give a minimal safe fix. Prefer complete files only if the user asks for code files.",
            "Do not rewrite architecture without need. Respect existing project structure.",
        ]
    elif task == "business":
        base += [
            "For business tasks: be practical. Give ready-to-use wording, risks, and the next action.",
        ]
    elif task == "analysis":
        base += [
            "For analysis: separate facts, assumptions, conclusions and next steps. Do not overstate confidence.",
        ]
    elif task == "writing":
        base += [
            "For writing tasks: produce polished user-ready text without explaining the writing process unless asked.",
        ]
    if web_requested and not web_ready:
        base += [
            "The user likely needs current data, but reliable live evidence was not available. Do not fabricate current facts.",
            "Say briefly that you cannot reliably verify current online data right now, then answer only with stable background information if useful.",
        ]
    return "\n".join(base)

def wants_live_web(text: str) -> bool:
    """Detect questions that need fresh / external facts.

    This is intentionally broad: users expect the bot to behave like a modern AI
    assistant. Static knowledge is fine for timeless explanations, but questions
    about people in office, news, prices, weather, schedules, laws, versions,
    ratings, companies, products and anything phrased as “now/today/latest” must
    go through live search.
    """
    t = (text or "").lower()
    if not t:
        return False

    current_markers = [
        "сегодня", "сейчас", "на данный момент", "актуаль", "последн", "новост",
        "курс", "цена", "стоимость", "сколько стоит", "расписан", "погода",
        "температура", "закон", "штраф", "правила", "кто сейчас", "кто президент",
        "кто премьер", "кто ceo", "кто глава", "выборы", "результаты", "матч",
        "релиз", "версия", "обновлен", "изменил", "рейтинг", "отзывы",
        "2026", "2025", "latest", "today", "now", "news", "current", "price",
        "schedule", "weather", "release", "version", "CEO",
    ]
    web_verbs = [
        "найди", "посмотри", "проверь", "поищи", "загугли", "в интернете",
        "источники", "ссылки", "research", "найди в сети", "проверь в сети",
    ]
    volatile_entities = [
        "трамп", "байден", "путин", "зеленск", "маск", "openai", "anthropic",
        "google", "apple", "telegram", "bitcoin", "битко", "доллар", "евро",
        "крипт", "акции", "нефть", "газ", "инфляц", "ставка",
    ]
    return (
        any(m in t for m in current_markers)
        or any(v in t for v in web_verbs)
        or any(e in t for e in volatile_entities)
    )


def web_is_configured() -> bool:
    return bool(TAVILY_API_KEY or SERPER_API_KEY or BRAVE_SEARCH_API_KEY)


def detect_weather_city(text: str) -> str:
    """Best-effort city detector for common Russian weather questions."""
    t = (text or "").lower().strip()
    if not any(word in t for word in ["погода", "температур", "weather"]):
        return ""

    city_patterns = [
        r"(?:в городе|в г\.?|в)\s+([а-яёa-z\- ]{2,40})",
        r"(?:для города|по городу)\s+([а-яёa-z\- ]{2,40})",
    ]
    city = ""
    for pattern in city_patterns:
        match = re.search(pattern, t, flags=re.I)
        if match:
            city = match.group(1).strip()
            break

    if not city:
        return ""

    city = re.sub(r"\b(какая|какой|сегодня|сейчас|завтра|россии|области|область|погода|температура)\b", " ", city)
    city = re.sub(r"[^а-яёa-z\- ]+", " ", city).strip()
    city = re.sub(r"\s+", " ", city)
    if not city:
        return ""

    # Normalize frequent Russian case forms for reliable external weather lookup.
    if "моск" in city:
        return "Moscow,Russia"
    if "киров" in city:
        return "Kirov,Russia"
    if "санкт" in city or "петербург" in city or "питер" in city:
        return "Saint Petersburg,Russia"
    if "казан" in city:
        return "Kazan,Russia"
    if "екатеринбург" in city:
        return "Yekaterinburg,Russia"
    if "новосибир" in city:
        return "Novosibirsk,Russia"

    return city



def _weather_description_ru(item: dict) -> str:
    values = item.get("lang_ru") or item.get("weatherDesc") or []
    if values and isinstance(values, list):
        return str(values[0].get("value") or "").strip()
    return ""


OPEN_METEO_WEATHER_RU = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "изморозь/туман",
    51: "слабая морось",
    53: "морось",
    55: "сильная морось",
    61: "слабый дождь",
    63: "дождь",
    65: "сильный дождь",
    71: "слабый снег",
    73: "снег",
    75: "сильный снег",
    80: "слабый ливень",
    81: "ливень",
    82: "сильный ливень",
    95: "гроза",
    96: "гроза с градом",
    99: "сильная гроза с градом",
}


def _weather_code_ru(code: Any) -> str:
    try:
        return OPEN_METEO_WEATHER_RU.get(int(code), "погодные условия уточняются")
    except Exception:
        return "погодные условия уточняются"


async def _open_meteo_weather(city: str) -> str:
    """Universal current weather via Open-Meteo, no API key required.

    This is intentionally used instead of generic web search for weather:
    search results often return unrelated pages/snippets, while weather requires
    a structured live-data endpoint.
    """
    geo = await _get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"Accept": "application/json", "User-Agent": "VSotahBot/1.0"},
        {"name": city, "count": 3, "language": "ru", "format": "json"},
    )
    results = geo.get("results") or []
    if not results:
        return ""

    # Prefer Russian locations when the city name is ambiguous.
    place = None
    for item in results:
        country_code = (item.get("country_code") or "").upper()
        if country_code == "RU":
            place = item
            break
    if place is None:
        place = results[0]

    lat = place.get("latitude")
    lon = place.get("longitude")
    if lat is None or lon is None:
        return ""

    data = await _get_json(
        "https://api.open-meteo.com/v1/forecast",
        {"Accept": "application/json", "User-Agent": "VSotahBot/1.0"},
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 1,
        },
    )
    current = data.get("current") or {}
    daily = data.get("daily") or {}
    if not current:
        return ""

    name = place.get("name") or city
    admin = place.get("admin1") or ""
    country = place.get("country") or ""
    location = ", ".join(x for x in [name, admin, country] if x)

    temp = current.get("temperature_2m")
    feels = current.get("apparent_temperature")
    humidity = current.get("relative_humidity_2m")
    wind = current.get("wind_speed_10m")
    desc = _weather_code_ru(current.get("weather_code"))

    max_list = daily.get("temperature_2m_max") or []
    min_list = daily.get("temperature_2m_min") or []
    rain_list = daily.get("precipitation_probability_max") or []
    day_extra = []
    if max_list and min_list:
        day_extra.append(f"днём примерно до {round(float(max_list[0]))}°C, минимум около {round(float(min_list[0]))}°C")
    if rain_list and rain_list[0] is not None:
        day_extra.append(f"вероятность осадков до {round(float(rain_list[0]))}%")

    lines = [f"Сейчас в {location}: {desc}."]
    if temp is not None:
        lines.append(f"Температура {round(float(temp))}°C")
    if feels is not None:
        lines.append(f"ощущается как {round(float(feels))}°C")
    if wind is not None:
        lines.append(f"ветер {round(float(wind))} км/ч")
    if humidity is not None:
        lines.append(f"влажность {round(float(humidity))}%")

    answer = "; ".join(lines) + "."
    if day_extra:
        answer += "\nНа сегодня: " + "; ".join(day_extra) + "."
    return answer + "\n\nДанные ориентировочные и могут быстро меняться."


async def fetch_weather_answer(query: str) -> str:
    city = detect_weather_city(query)
    if not city:
        return ""

    try:
        answer = await _open_meteo_weather(city)
        if answer:
            return answer
    except Exception as e:
        print(f"OPEN METEO WEATHER ERROR: {short_error_text(e)}")

    # Fallback to wttr.in if Open-Meteo is temporarily unavailable.
    try:
        data = await _get_json(
            f"https://wttr.in/{city}",
            {"Accept": "application/json", "User-Agent": "VSotahBot/1.0"},
            {"format": "j1", "lang": "ru"},
        )
        current = (data.get("current_condition") or [{}])[0]
        temp = current.get("temp_C")
        feels = current.get("FeelsLikeC")
        humidity = current.get("humidity")
        wind = current.get("windspeedKmph")
        desc = _weather_description_ru(current)
        parts = []
        if desc:
            parts.append(desc.lower())
        if temp not in {None, ""}:
            parts.append(f"температура {temp}°C")
        if feels not in {None, ""}:
            parts.append(f"ощущается как {feels}°C")
        if wind not in {None, ""}:
            parts.append(f"ветер {wind} км/ч")
        if humidity not in {None, ""}:
            parts.append(f"влажность {humidity}%")
        if parts:
            return f"Сейчас погода: {city}. " + "; ".join(parts) + "."
    except Exception as e:
        print(f"WTTR WEATHER ERROR: {short_error_text(e)}")
    return ""


def detect_crypto_symbols(text: str) -> list[tuple[str, str]]:
    """Detect common crypto price questions and return CoinGecko ids with display names."""
    t = (text or "").lower()
    pairs = [
        ("bitcoin", "Bitcoin", ["bitcoin", "биткоин", "биткойн", "btc", "бтс"]),
        ("ethereum", "Ethereum", ["ethereum", "эфир", "эфириум", "eth"]),
        ("solana", "Solana", ["solana", "солана", "sol"]),
        ("toncoin", "Toncoin", ["toncoin", "ton", "тонкоин", "тон"]),
        ("binancecoin", "BNB", ["bnb", "бинанс", "binance coin"]),
        ("ripple", "XRP", ["xrp", "ripple", "рипл"]),
        ("dogecoin", "Dogecoin", ["doge", "dogecoin", "догикоин"]),
    ]
    found: list[tuple[str, str]] = []
    price_words = ["цена", "стоимость", "курс", "сколько стоит", "price", "cost", "rate"]
    if not any(w in t for w in price_words):
        return []
    for coin_id, name, aliases in pairs:
        if any(alias in t for alias in aliases):
            found.append((coin_id, name))
    return found[:4]


BINANCE_SYMBOLS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "solana": "SOLUSDT",
    "toncoin": "TONUSDT",
    "binancecoin": "BNBUSDT",
    "ripple": "XRPUSDT",
    "dogecoin": "DOGEUSDT",
}


def _fmt_money(value: Any, currency: str = "USD") -> str:
    try:
        number = float(value)
    except Exception:
        return "н/д"
    if currency == "USD":
        return f"${number:,.2f}"
    if currency == "RUB":
        return f"{number:,.0f} ₽".replace(",", " ")
    if currency == "EUR":
        return f"€{number:,.2f}"
    return f"{number:,.2f}"


async def fetch_crypto_price_answer(query: str) -> str:
    """Return a direct user-ready crypto price answer without waiting for an LLM.

    This makes simple questions like “Какая сейчас цена биткоина?” fast and reliable.
    It first tries CoinGecko for USD/RUB/EUR and falls back to Binance USD ticker.
    """
    coins = detect_crypto_symbols(query)
    if not coins:
        return ""

    ids = ",".join(coin_id for coin_id, _ in coins)

    # 1) CoinGecko: gives USD/RUB/EUR and 24h change without an API key.
    try:
        data = await _get_json(
            "https://api.coingecko.com/api/v3/simple/price",
            {"Accept": "application/json"},
            {
                "ids": ids,
                "vs_currencies": "usd,rub,eur",
                "include_24hr_change": "true",
                "include_last_updated_at": "true",
            },
        )
        lines = ["📈 Актуальная цена криптовалюты:\n"]
        has_price = False
        for coin_id, name in coins:
            item = data.get(coin_id) or {}
            if not item.get("usd"):
                continue
            has_price = True
            change = item.get("usd_24h_change")
            change_text = ""
            if change is not None:
                sign = "+" if float(change) >= 0 else ""
                change_text = f"\n• Изменение за 24ч: {sign}{float(change):.2f}%"
            lines.append(
                f"• {name}: {_fmt_money(item.get('usd'), 'USD')}"
                f" / {_fmt_money(item.get('rub'), 'RUB')}"
                f" / {_fmt_money(item.get('eur'), 'EUR')}"
                f"{change_text}"
            )
        if has_price:
            lines.append("\nЦена ориентировочная и может быстро меняться. Источник: CoinGecko.")
            return "\n".join(lines)
    except Exception as e:
        print(f"CRYPTO DIRECT COINGECKO ERROR: {short_error_text(e)}")

    # 2) Binance fallback: very fast for USD/USDT quotes.
    lines = ["📈 Актуальная цена криптовалюты:\n"]
    has_price = False
    for coin_id, name in coins:
        symbol = BINANCE_SYMBOLS.get(coin_id)
        if not symbol:
            continue
        try:
            data = await _get_json(
                "https://api.binance.com/api/v3/ticker/24hr",
                {"Accept": "application/json"},
                {"symbol": symbol},
            )
            price = data.get("lastPrice")
            change = data.get("priceChangePercent")
            if not price:
                continue
            has_price = True
            sign = "+" if change is not None and float(change) >= 0 else ""
            lines.append(
                f"• {name}: {_fmt_money(price, 'USD')}"
                f"\n• Изменение за 24ч: {sign}{float(change):.2f}%"
            )
        except Exception as e:
            print(f"CRYPTO DIRECT BINANCE ERROR {symbol}: {short_error_text(e)}")
    if has_price:
        lines.append("\nЦена ориентировочная и может быстро меняться. Источник: Binance.")
        return "\n".join(lines)

    return ""


async def fetch_crypto_price_context(query: str) -> str:
    coins = detect_crypto_symbols(query)
    if not coins:
        return ""
    ids = ",".join(coin_id for coin_id, _ in coins)
    try:
        data = await _get_json(
            "https://api.coingecko.com/api/v3/simple/price",
            {"Accept": "application/json"},
            {
                "ids": ids,
                "vs_currencies": "usd,rub,eur",
                "include_24hr_change": "true",
                "include_last_updated_at": "true",
            },
        )
        lines = [
            "LIVE MARKET CONTEXT. Use this for current crypto prices. Source: CoinGecko simple price API. "
            "Tell the user prices are approximate and can change quickly."
        ]
        for coin_id, name in coins:
            item = data.get(coin_id) or {}
            if not item:
                continue
            usd = item.get("usd")
            rub = item.get("rub")
            eur = item.get("eur")
            change = item.get("usd_24h_change")
            updated = item.get("last_updated_at")
            line = f"{name}: USD={usd}; RUB={rub}; EUR={eur}; 24h_change_usd_percent={change}; last_updated_unix={updated}"
            lines.append(line)
        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception as e:
        print(f"CRYPTO PRICE ERROR: {short_error_text(e)}")
        return ""


async def _post_json(url: str, headers: dict, payload: dict) -> dict:
    timeout = aiohttp.ClientTimeout(total=WEB_SEARCH_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as response:
            response.raise_for_status()
            return await response.json()


async def _get_json(url: str, headers: dict, params: dict) -> dict:
    timeout = aiohttp.ClientTimeout(total=WEB_SEARCH_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers, params=params) as response:
            response.raise_for_status()
            return await response.json()


def infer_tavily_topic(query: str) -> str:
    t = (query or "").lower()
    finance_markers = [
        "цена", "курс", "стоимость", "акции", "биржа", "крипт", "битко", "bitcoin",
        "btc", "ethereum", "eth", "доллар", "евро", "рубль", "market", "stock", "price", "finance",
    ]
    news_markers = ["новост", "сегодня", "последн", "событ", "что произошло", "latest", "news", "today"]
    if any(x in t for x in finance_markers):
        return "finance"
    if any(x in t for x in news_markers):
        return "news"
    return "general"


def infer_tavily_time_range(query: str) -> str | None:
    t = (query or "").lower()
    if any(x in t for x in ["сегодня", "сейчас", "за день", "24ч", "today", "now", "latest"]):
        return "day"
    if any(x in t for x in ["недел", "за неделю", "week"]):
        return "week"
    if any(x in t for x in ["месяц", "month"]):
        return "month"
    return None


async def _post_tavily(payload: dict) -> dict:
    """Call Tavily with several auth styles for compatibility across accounts."""
    if not TAVILY_API_KEY:
        return {}

    auth_attempts = [
        ({"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"}, payload),
        ({"X-API-Key": TAVILY_API_KEY, "Content-Type": "application/json"}, payload),
        ({"Content-Type": "application/json"}, {**payload, "api_key": TAVILY_API_KEY}),
    ]

    last_error: Exception | None = None
    for headers, body in auth_attempts:
        try:
            return await _post_json("https://api.tavily.com/search", headers, body)
        except Exception as e:
            last_error = e
            continue
    if last_error:
        raise last_error
    return {}


async def search_web(query: str) -> list[dict]:
    """Returns normalized search results: title, url, snippet, score.

    Important: this function does NOT create a final user answer. It only gathers
    evidence. The LLM must then reason over the evidence and refuse to invent facts
    if evidence is weak or irrelevant.
    """
    user_query = normalize_search_query(query)
    search_query = build_search_query(query)
    if not search_query or not WEB_SEARCH_ENABLED:
        return []

    try:
        if TAVILY_API_KEY:
            payload = {
                "query": search_query,
                "search_depth": "advanced",
                "topic": infer_tavily_topic(user_query),
                "max_results": max(WEB_SEARCH_MAX_RESULTS, 6),
                "include_answer": True,
                "include_raw_content": False,
                "include_images": False,
                "include_favicon": False,
            }
            time_range = infer_tavily_time_range(user_query)
            if time_range:
                payload["time_range"] = time_range

            data = await _post_tavily(payload)

            normalized: list[dict] = []
            answer = (data.get("answer") or "").strip()
            if answer:
                normalized.append({
                    "title": "Поисковая сводка",
                    "url": "",
                    "snippet": answer,
                    "score": 1.0,
                    "kind": "answer",
                })

            for item in data.get("results", [])[:max(WEB_SEARCH_MAX_RESULTS, 6)]:
                url = item.get("url") or ""
                content = item.get("content") or item.get("snippet") or ""
                title = item.get("title") or "Источник"
                score = item.get("score")
                if not url and not content:
                    continue
                normalized.append({
                    "title": title,
                    "url": url,
                    "snippet": content,
                    "score": score,
                    "kind": "source",
                })
            return normalized[:7]

        if SERPER_API_KEY:
            data = await _post_json(
                "https://google.serper.dev/search",
                {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                {"q": search_query, "num": max(WEB_SEARCH_MAX_RESULTS, 6), "hl": "ru"},
            )
            organic = data.get("organic", []) or []
            return [
                {
                    "title": item.get("title") or "Источник",
                    "url": item.get("link") or "",
                    "snippet": item.get("snippet") or "",
                    "score": None,
                    "kind": "source",
                }
                for item in organic[:max(WEB_SEARCH_MAX_RESULTS, 6)]
                if item.get("link")
            ]

        if BRAVE_SEARCH_API_KEY:
            data = await _get_json(
                "https://api.search.brave.com/res/v1/web/search",
                {"X-Subscription-Token": BRAVE_SEARCH_API_KEY, "Accept": "application/json"},
                {"q": search_query, "count": max(WEB_SEARCH_MAX_RESULTS, 6)},
            )
            results = ((data.get("web") or {}).get("results") or [])[:max(WEB_SEARCH_MAX_RESULTS, 6)]
            return [
                {
                    "title": item.get("title") or "Источник",
                    "url": item.get("url") or "",
                    "snippet": item.get("description") or "",
                    "score": None,
                    "kind": "source",
                }
                for item in results
                if item.get("url")
            ]
    except Exception as e:
        print(f"WEB SEARCH ERROR: {short_error_text(e)}")
        return []

    return []

def render_web_context(results: list[dict]) -> str:
    if not results:
        return ""

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f"LIVE WEB EVIDENCE. Current UTC date: {today}.",
        "Use this evidence for fresh facts. The evidence may contain irrelevant search noise; check relevance before answering.",
        "Critical rules:",
        "1) Answer ONLY in Russian unless the user explicitly asks another language.",
        "2) Do not expose raw search snippets, JSON, API names, Railway, keys, admin settings or internal tools.",
        "3) Do not invent current facts if evidence is weak or unrelated.",
        "4) For current facts, use wording like 'по найденным данным' and keep the answer concise.",
        "5) Sources are optional: add at most 2 short source lines only if they are clearly relevant.",
    ]
    answer_items = [item for item in results if item.get("kind") == "answer"]
    source_items = [item for item in results if item.get("kind") != "answer"]
    for item in answer_items[:1]:
        snippet = (item.get("snippet") or "").strip()
        if snippet:
            lines.append(f"SEARCH_SUMMARY:\n{snippet[:1200]}")
    for idx, item in enumerate(source_items[:6], start=1):
        title = (item.get("title") or "Источник").strip()
        url = (item.get("url") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        score = item.get("score")
        lines.append(f"SOURCE {idx}\nTitle: {title}\nURL: {url}\nScore: {score}\nText: {snippet[:1100]}")
    return "\n\n".join(lines)[:9000]


def build_web_quality_instruction(query: str, results: list[dict]) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        f"Пользователь задал вопрос, которому могут быть нужны свежие данные. Сегодня {today} UTC.\n"
        f"Вопрос пользователя: {normalize_search_query(query)}\n\n"
        "Ответь как умный ассистент: по-русски, прямо, без технических деталей. "
        "Если источники не отвечают на вопрос или выглядят нерелевантными, НЕ фантазируй: "
        "скажи, что не удалось надежно проверить актуальные данные, и предложи, что именно уточнить. "
        "Не вставляй длинные ссылки в основной текст. Если источники полезны, максимум 2 строки в конце: 'Источники: ...'."
    )

def render_sources(results: list[dict], max_sources: int = 3) -> str:
    sources = [item for item in results if item.get("kind") != "answer" and item.get("url")]
    if not sources:
        return ""
    lines = ["Источники:"]
    for item in sources[:max_sources]:
        title = (item.get("title") or "Источник").strip()
        url = (item.get("url") or "").strip()
        lines.append(f"• {title}: {url}")
    return "\n".join(lines)


def build_direct_web_answer(query: str, results: list[dict]) -> str:
    """Direct answers are only allowed for deterministic data endpoints.

    Generic Tavily results must go through the LLM with strict evidence rules.
    Returning raw search snippets directly made answers look like random links.
    """
    return ""


async def direct_live_answer(query: str, force_web: bool = False) -> str:
    query = normalize_search_query(query)
    if not query:
        return ""

    # Direct deterministic tools are fine. Generic web search is handled by the LLM.
    weather_answer = await fetch_weather_answer(query)
    if weather_answer:
        return weather_answer

    crypto_answer = await fetch_crypto_price_answer(query)
    if crypto_answer:
        return crypto_answer

    return ""

def mode_instruction(mode: str) -> str:
    if mode == "research":
        return (
            "Ты Deep Research Lite внутри Telegram. Дай структурированный отчёт: краткий вывод, факты, "
            "практические выводы, риски/ограничения и источники, если они есть. Не растягивай без пользы."
        )
    if mode == "business":
        return (
            "Ты бизнес-ассистент. Помогай с КП, письмами, договорами, продажами, стратегией, таблицами, "
            "операционкой и анализом. Ответ должен быть практичным, с готовыми формулировками и следующим шагом."
        )
    if mode == "code":
        return (
            "Ты senior code assistant. Помогай с Python, Telegram bots, Railway, GitHub и API. "
            "Давай готовые файлы/команды, объясняй ошибки коротко и точно, не ломай существующую архитектуру."
        )
    if mode == "web":
        return (
            "Ты AI с live web context. Сначала используй найденный web-контекст, потом свои выводы. "
            "Отвечай прямо и полезно. Если есть источники — добавь короткий блок 'Источники'. "
            "Никогда не говори пользователю про API key, админа, Railway или настройки."
        )
    return ""


async def enrich_messages_with_web(messages: list[dict], force_web: bool = False) -> tuple[list[dict], bool, bool]:
    query = latest_user_text(messages)
    should_search = force_web or wants_live_web(query)
    if not should_search:
        return messages, False, False

    # For crypto prices, use a direct market-data endpoint first. It is faster and more precise than generic search.
    crypto_context = await fetch_crypto_price_context(query)
    if crypto_context:
        return [{"role": "system", "content": crypto_context}, *messages], True, True

    if not web_is_configured():
        notice = (
            "LIVE WEB SEARCH REQUESTED, BUT ONLINE SEARCH IS NOT AVAILABLE IN THIS RUNTIME. "
            "Do not mention API keys, admin settings, Tavily, Serper, Brave, Railway, or configuration to the user. "
            "Give a helpful normal answer if possible. If fresh facts are essential, say briefly in Russian: "
            "'Сейчас я не могу проверить актуальные онлайн-данные.'"
        )
        return [{"role": "system", "content": notice}, *messages], True, False

    results = await search_web(query)
    context = render_web_context(results)
    if not context:
        return [
            {
                "role": "system",
                "content": (
                    "Live web search returned no usable results. Do not mention APIs, admin settings, Tavily, Railway, or keys. "
                    "If current data is required, answer in Russian: 'Не удалось надежно проверить актуальные данные прямо сейчас.' "
                    "Then give only safe general guidance, clearly separated from verified current facts."
                ),
            },
            *messages,
        ], True, False
    quality = build_web_quality_instruction(query, results)
    return [{"role": "system", "content": quality + "\n\n" + context}, *messages], True, True

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


async def ai_router(selected_model: str, messages: list[dict], mode: str = "chat", force_web: bool = False) -> str:
    mode = (mode or "chat").lower()

    # Fast direct live answer for weather/crypto/current web questions.
    # This prevents stale LLM replies like “I do not have access”.
    direct_answer = await direct_live_answer(
        latest_user_text(messages),
        force_web=force_web or mode in {"web", "research"},
    )
    if direct_answer:
        return direct_answer

    messages, web_requested, web_ready = await enrich_messages_with_web(messages, force_web=force_web or mode in {"web", "research"})

    # Important: web context is stored as system messages. OpenAI can consume these directly,
    # but Claude/Gemini paths use a separate system prompt. Move system-context into
    # base_system so every provider receives the live web data.
    system_contexts = [
        (msg.get("content") or "").strip()
        for msg in messages
        if msg.get("role") == "system" and (msg.get("content") or "").strip()
    ]
    dialogue_messages = [msg for msg in messages if msg.get("role") != "system"]
    memory = build_memory(dialogue_messages, limit=TEXT_HISTORY_LIMIT)

    base_system = system_prompt(get_personality(selected_model))
    extra = mode_instruction(mode)
    if extra:
        base_system = f"{base_system}\n\n{extra}"
    quality = quality_layer_instruction(latest_user_text(messages), mode, web_requested=web_requested, web_ready=web_ready)
    base_system = f"{base_system}\n\n{quality}"
    if system_contexts:
        base_system = f"{base_system}\n\n" + "\n\n".join(system_contexts[-3:])
    if web_ready:
        base_system = (
            f"{base_system}\n\n"
            "AI BRAIN 2.0 STRICT WEB MODE: Ответ должен быть на русском, ясный и человеческий. "
            "Используй live web evidence только если оно действительно отвечает на вопрос. "
            "Не копируй сырые snippets. Не меняй язык ответа. Не вставляй длинные URL в середину текста. "
            "Если evidence противоречивое или не по теме — честно скажи, что не удалось надежно проверить актуальные данные. "
            "Никогда не упоминай API, Tavily, ключи, админа, Railway или внутреннюю реализацию."
        )
    # модель должна честно сказать об этом, а не выдумывать свежие факты.
    if web_requested and not web_ready and (mode in {"web", "research"} or wants_live_web(latest_user_text(messages))):
        base_system = (
            f"{base_system}\n\n"
            "Важное правило: пользователь просит актуальные/live данные, но актуальный онлайн-поиск сейчас недоступен или не дал результатов. "
            "Не придумывай свежие новости, цены, расписания или законы. Не упоминай API, ключи, администратора, Railway или технические настройки. "
            "Ответь полезно обычным языком; если без свежих данных нельзя, коротко скажи: сейчас я не могу проверить актуальные онлайн-данные."
        )

    openai_messages = normalize_openai_messages([{"role": "system", "content": base_system}, *memory])

    max_tokens = TEXT_MAX_TOKENS
    if mode == "research":
        max_tokens = RESEARCH_MAX_TOKENS
    elif mode == "business":
        max_tokens = BUSINESS_MAX_TOKENS
    elif mode == "code":
        max_tokens = CODE_MAX_TOKENS

    calls = {
        "openai": lambda: openai_chat(openai_messages, OPENAI_TEXT_MODEL, max_tokens, 0.35 if mode in {"research", "code"} else 0.45),
        "anthropic": lambda: anthropic_chat(base_system, memory, ANTHROPIC_TEXT_MODEL, max_tokens, 0.35 if mode in {"research", "code"} else 0.45),
        "gemini": lambda: gemini_chat(base_system, memory, GEMINI_TEXT_MODEL),
        "deepseek": lambda: deepseek_chat(openai_messages, max_tokens, 0.35 if mode in {"research", "code"} else 0.45),
    }
    return await run_with_fallback(provider_order(selected_model, "text"), calls, f"{mode}:{selected_model}")


async def research_router(selected_model: str, messages: list[dict]) -> str:
    return await ai_router(selected_model, messages, mode="research", force_web=True)


async def business_router(selected_model: str, messages: list[dict]) -> str:
    return await ai_router(selected_model, messages, mode="business", force_web=False)


async def code_router(selected_model: str, messages: list[dict]) -> str:
    return await ai_router(selected_model, messages, mode="code", force_web=False)


async def web_router(selected_model: str, messages: list[dict]) -> str:
    return await ai_router(selected_model, messages, mode="web", force_web=True)


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




