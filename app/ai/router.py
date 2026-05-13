import asyncio
import base64
import json
import os
import re
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

TEXT_HISTORY_LIMIT = int(os.getenv("TEXT_HISTORY_LIMIT", "6"))
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


def wants_live_web(text: str) -> bool:
    """Lightweight detector for current-data questions. No web call without provider API key."""
    t = (text or "").lower()
    current_markers = [
        "сегодня", "сейчас", "актуаль", "последн", "новост", "курс", "цена", "стоимость",
        "расписан", "погода", "закон", "2026", "2025", "latest", "today", "now", "news",
        "current", "price", "schedule", "weather", "release", "обновлен", "изменил",
    ]
    web_verbs = ["найди", "посмотри", "проверь", "поищи", "загугли", "в интернете", "источники", "ссылки", "research"]
    return any(m in t for m in current_markers) or any(v in t for v in web_verbs)


def web_is_configured() -> bool:
    return bool(TAVILY_API_KEY or SERPER_API_KEY or BRAVE_SEARCH_API_KEY)


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


async def search_web(query: str) -> list[dict]:
    """Returns normalized search results: title, url, snippet.

    Tavily is the primary provider. The current Tavily API requires Bearer auth;
    keeping the API key only in JSON silently fails on newer accounts.
    """
    query = normalize_search_query(query)
    if not query or not WEB_SEARCH_ENABLED:
        return []

    try:
        if TAVILY_API_KEY:
            payload = {
                "query": query,
                "search_depth": "fast",
                "topic": infer_tavily_topic(query),
                "max_results": WEB_SEARCH_MAX_RESULTS,
                "include_answer": "basic",
                "include_raw_content": False,
                "include_images": False,
                "include_favicon": False,
            }
            time_range = infer_tavily_time_range(query)
            if time_range:
                payload["time_range"] = time_range

            data = await _post_json(
                "https://api.tavily.com/search",
                {
                    "Authorization": f"Bearer {TAVILY_API_KEY}",
                    "Content-Type": "application/json",
                },
                payload,
            )

            normalized: list[dict] = []
            answer = (data.get("answer") or "").strip()
            if answer:
                normalized.append({
                    "title": "Tavily краткий ответ",
                    "url": "",
                    "snippet": answer,
                    "kind": "answer",
                })

            for item in data.get("results", [])[:WEB_SEARCH_MAX_RESULTS]:
                url = item.get("url") or ""
                content = item.get("content") or item.get("snippet") or ""
                if not url and not content:
                    continue
                normalized.append({
                    "title": item.get("title") or "Источник",
                    "url": url,
                    "snippet": content,
                    "score": item.get("score"),
                    "kind": "source",
                })
            return normalized

        if SERPER_API_KEY:
            data = await _post_json(
                "https://google.serper.dev/search",
                {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                {"q": query, "num": WEB_SEARCH_MAX_RESULTS},
            )
            organic = data.get("organic", []) or []
            return [
                {
                    "title": item.get("title") or "Источник",
                    "url": item.get("link") or "",
                    "snippet": item.get("snippet") or "",
                }
                for item in organic[:WEB_SEARCH_MAX_RESULTS]
                if item.get("link")
            ]

        if BRAVE_SEARCH_API_KEY:
            data = await _get_json(
                "https://api.search.brave.com/res/v1/web/search",
                {"X-Subscription-Token": BRAVE_SEARCH_API_KEY, "Accept": "application/json"},
                {"q": query, "count": WEB_SEARCH_MAX_RESULTS},
            )
            results = ((data.get("web") or {}).get("results") or [])[:WEB_SEARCH_MAX_RESULTS]
            return [
                {
                    "title": item.get("title") or "Источник",
                    "url": item.get("url") or "",
                    "snippet": item.get("description") or "",
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
    lines = [
        "LIVE WEB CONTEXT FROM TAVILY/SEARCH. Use this context for актуальные данные.",
        "Answer in the user's language. Do not mention API keys, Railway, admin settings, or internal tools.",
        "When relying on sources, add a short 'Источники:' section with source titles and URLs.",
    ]
    answer_items = [item for item in results if item.get("kind") == "answer"]
    source_items = [item for item in results if item.get("kind") != "answer"]
    for item in answer_items[:1]:
        snippet = (item.get("snippet") or "").strip()
        if snippet:
            lines.append(f"TAVILY_SHORT_ANSWER:\n{snippet}")
    for idx, item in enumerate(source_items, start=1):
        title = (item.get("title") or "Источник").strip()
        url = (item.get("url") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        lines.append(f"[{idx}] {title}\nURL: {url}\nSnippet: {snippet}")
    return "\n\n".join(lines)[:7500]


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
                    "Live web search returned no usable results. Do not mention APIs or admin settings. "
                    "If current data is required, say briefly that online results were not found and give the best general guidance."
                ),
            },
            *messages,
        ], True, False
    return [{"role": "system", "content": context}, *messages], True, True

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

    # Fast direct market-data answer for simple crypto price questions.
    # This avoids vague LLM responses like “I do not have access”.
    direct_crypto_answer = await fetch_crypto_price_answer(latest_user_text(messages))
    if direct_crypto_answer:
        return direct_crypto_answer

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
    if system_contexts:
        base_system = f"{base_system}\n\n" + "\n\n".join(system_contexts[-3:])
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




