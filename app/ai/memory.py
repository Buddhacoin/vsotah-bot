import re
from collections import deque


MAX_MEMORY_NOTE_CHARS = 2200
MAX_MEMORY_ITEM_CHARS = 260


_NOISE_PREFIXES = (
    "/start", "/health", "/admin", "/stats", "/payments", "/users", "/errors",
    "/deletecontext", "/premium", "/models", "/channels", "/referral",
)

_MEMORY_MARKERS = (
    "запомни", "помни", "важно", "мне нужно", "мне надо", "я хочу", "я делаю",
    "мой проект", "наш проект", "предпочитаю", "лучше", "не надо", "не трогай",
    "работаем над", "цель", "план", "задача", "ошибка", "исправить",
)

_PREFERENCE_MARKERS = (
    "предпочитаю", "лучше", "не надо", "не трогай", "делай", "пиши", "присылай",
    "полными файлами", "zip", "без патчей", "не ломай", "не откатывай",
)

_PROJECT_MARKERS = (
    "vsotah", "бот", "railway", "github", "telegram", "ai", "openai", "claude",
    "gemini", "tavily", "referral", "webhook", "voice", "файл", "модель",
)


def _clean_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_noise(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return True
    if any(low.startswith(x) for x in _NOISE_PREFIXES):
        return True
    if low.startswith("[") and "]" in low[:35]:
        return True
    if len(low) < 8:
        return True
    return False


def _clip(text: str, limit: int = MAX_MEMORY_ITEM_CHARS) -> str:
    text = _clean_text(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _dedupe_keep_order(items: list[str], limit: int = 8) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = re.sub(r"[^а-яa-z0-9]+", " ", item.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def build_dialogue_memory_note(messages: list[dict], max_items: int = 10) -> str:
    """Build a compact deterministic memory note from recent dialogue.

    This is intentionally not a second LLM call: it is cheap, safe and cannot
    hallucinate. The note helps the main model keep continuity in long chats.
    """
    if not messages:
        return ""

    user_texts: deque[str] = deque(maxlen=40)
    assistant_texts: deque[str] = deque(maxlen=10)
    for msg in messages:
        role = msg.get("role")
        content = _clean_text(msg.get("content") or "")
        if not content or _is_noise(content):
            continue
        if role == "user":
            user_texts.append(content)
        elif role == "assistant":
            assistant_texts.append(content)

    facts: list[str] = []
    preferences: list[str] = []
    active_topics: list[str] = []

    for text in reversed(user_texts):
        low = text.lower()
        if any(marker in low for marker in _PREFERENCE_MARKERS):
            preferences.append(_clip(text))
        if any(marker in low for marker in _MEMORY_MARKERS):
            facts.append(_clip(text))
        if any(marker in low for marker in _PROJECT_MARKERS):
            active_topics.append(_clip(text, 180))

    facts = _dedupe_keep_order(facts, limit=max_items)
    preferences = _dedupe_keep_order(preferences, limit=6)
    active_topics = _dedupe_keep_order(active_topics, limit=6)

    parts: list[str] = []
    if facts:
        parts.append("Важные факты/задачи пользователя:\n" + "\n".join(f"• {x}" for x in facts))
    if preferences:
        parts.append("Предпочтения пользователя по работе:\n" + "\n".join(f"• {x}" for x in preferences))
    if active_topics:
        parts.append("Недавний рабочий контекст:\n" + "\n".join(f"• {x}" for x in active_topics))

    if not parts:
        return ""

    note = (
        "AI MEMORY 2.0 — краткая память диалога. Используй это только как контекст, "
        "не цитируй дословно и не выдумывай факты.\n\n" + "\n\n".join(parts)
    )
    return note[:MAX_MEMORY_NOTE_CHARS]


def build_memory(messages: list[dict], limit: int = 12) -> list[dict]:
    """
    Собирает короткую память диалога для быстрой и стабильной работы.
    Оставляет последние сообщения, убирает пустые и слишком длинные куски.
    System memory/context messages are preserved separately at the beginning.
    """
    if not messages:
        return []

    system_items = []
    dialogue = []
    for msg in messages:
        role = msg.get("role", "user")
        content = _clean_text(msg.get("content") or "")

        if role not in {"user", "assistant", "system"}:
            continue
        if not content:
            continue

        item = {"role": role, "content": content[:4000]}
        if role == "system":
            system_items.append(item)
        else:
            dialogue.append(item)

    recent = dialogue[-limit:]
    return system_items[-3:] + recent
