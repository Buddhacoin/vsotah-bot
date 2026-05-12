from dataclasses import dataclass
from typing import Literal

VisionKind = Literal["screenshot", "document", "receipt", "ui", "photo", "art", "meme", "logo", "unknown"]


@dataclass(frozen=True)
class VisionPlan:
    kind: VisionKind
    instruction: str


def _text(value: str | None) -> str:
    return (value or "").strip()


def _lower(value: str | None) -> str:
    return _text(value).lower()


def _has_any(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)


def detect_vision_kind(user_text: str | None = None, image_context: str | None = None) -> VisionKind:
    text = f"{_lower(user_text)} {_lower(image_context)}"

    if _has_any(text, ["скрин", "screenshot", "экран", "ошибка", "лог", "railway", "github"]):
        return "screenshot"
    if _has_any(text, ["чек", "receipt", "инвойс", "invoice", "квитанц"]):
        return "receipt"
    if _has_any(text, ["документ", "паспорт", "pdf", "таблица", "document"]):
        return "document"
    if _has_any(text, ["интерфейс", "ui", "ux", "приложение", "сайт", "кнопка"]):
        return "ui"
    if _has_any(text, ["мем", "meme"]):
        return "meme"
    if _has_any(text, ["логотип", "лого", "logo"]):
        return "logo"
    if _has_any(text, ["арт", "рисунок", "иллюстрац", "anime", "painting"]):
        return "art"
    if _has_any(text, ["фото", "изображено", "кто", "что на", "photo", "picture"]):
        return "photo"

    return "unknown"


def build_vision_plan(user_text: str | None = None, image_context: str | None = None) -> VisionPlan:
    kind = detect_vision_kind(user_text, image_context)
    instructions = {
        "screenshot": (
            "This is likely a screenshot. Read visible interface text carefully, identify errors, buttons, status labels, "
            "and explain exactly what the user should do next. Be practical and step-by-step when needed."
        ),
        "document": (
            "This may be a document. Extract visible text, summarize the key points, and answer the user's question. "
            "If something is unreadable, say what is unclear instead of inventing details."
        ),
        "receipt": (
            "This may be a receipt or invoice. Extract amounts, dates, merchant names, payment status, and important fields."
        ),
        "ui": (
            "This may be an app or website interface. Describe layout, UX problems, buttons, and practical improvements."
        ),
        "photo": (
            "Describe the photo accurately, focusing on objects, scene, context, quality, and the user's specific question."
        ),
        "art": (
            "Analyze the visual style, composition, details, and how to improve or recreate it if requested."
        ),
        "meme": (
            "Explain the meme or joke carefully and identify visible text, but do not over-explain if the answer can be simple."
        ),
        "logo": (
            "Analyze the logo shape, readability, uniqueness, balance, and how to improve it for brand use."
        ),
        "unknown": (
            "Analyze the image accurately and answer the user's question. Mention uncertainty when details are not visible."
        ),
    }
    return VisionPlan(kind=kind, instruction=instructions.get(kind, instructions["unknown"]))


def build_vision_prompt(user_text: str | None, image_context: str | None = None) -> str:
    request = _text(user_text) or "Что изображено на фото? Опиши подробно и помоги пользователю."
    plan = build_vision_plan(request, image_context)

    prompt = (
        "Ты Vision AI внутри VSotahBot. Отвечай на русском, точно и полезно.\n"
        "Не выдумывай невидимые детали. Если текст или мелкие элементы плохо читаются — скажи об этом.\n\n"
        f"Тип изображения: {plan.kind}\n"
        f"Инструкция анализа: {plan.instruction}\n\n"
        f"Запрос пользователя: {request}"
    )

    if image_context:
        prompt += f"\n\nДополнительный контекст: {image_context}"

    return prompt[:3000]


def build_pdf_vision_prompt(filename: str, question: str | None) -> str:
    request = _text(question) or "Проанализируй PDF, прочитай видимый текст и выдели главное."
    return (
        f"Пользователь отправил PDF-файл: {filename}.\n"
        "PDF не содержит обычного текстового слоя или плохо читается как текст, поэтому ниже страницы как изображения.\n\n"
        f"Задача пользователя: {request}\n\n"
        "Прочитай всё, что видно на страницах. Сохраняй факты, суммы, даты, имена и важные поля. "
        "Если часть текста неразборчива, честно отметь это. Ответь по делу."
    )[:3000]

