from datetime import datetime
import re


def system_prompt(personality: str = "") -> str:
    current_datetime = datetime.now().strftime("%d.%m.%Y %H:%M")
    today_text = datetime.now().strftime("%A, %d.%m.%Y")

    base = (
        "Ты профессиональный AI-ассистент внутри Telegram. "
        "Отвечай на языке пользователя. Если пользователь пишет на русском — отвечай на русском. "
        "Оформляй ответы чисто и спокойно, чтобы их было удобно читать в Telegram. "
        "Не используй markdown-таблицы, заголовки с ###, цитаты через >, декоративные линии, лишние разделители, много эмодзи или визуальный мусор. "
        "Не перегружай ответ символами ✅ ❌ 🤝 и подобным, если пользователь сам не просит красивый пост. "
        "Лучший формат: короткие абзацы, понятные пункты только когда они реально нужны. "
        "Если пользователь просит сравнение — сравни обычным текстом, без таблицы. "
        "Если пользователь отправил изображение, внимательно проанализируй его и ответь по вопросу. "
        "Не повторяй одно и то же. Не растягивай ответ без необходимости. "
        f"Текущая дата и время: {current_datetime}. Сегодня: {today_text}."
    )

    if personality:
        return f"{base}\n\nСтиль выбранной модели:\n{personality}"
    return base


def clean_ai_answer(text: str) -> str:
    """Убирает визуальный мусор, который плохо выглядит в Telegram."""
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*>\s?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*_]{3,}\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[`*_]{2,}", "", cleaned)

    lines = cleaned.split("\n")
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"[|\-:\s]+", stripped):
            continue
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            cells = [c.strip() for c in stripped.strip("|").split("|") if c.strip()]
            if cells:
                new_lines.append(" — ".join(cells))
        else:
            new_lines.append(line.rstrip())

    cleaned = "\n".join(new_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()

