import re
from dataclasses import dataclass
from typing import Literal

ImageKind = Literal[
    "avatar",
    "logo",
    "photo",
    "art",
    "ui",
    "meme",
    "document",
    "product",
    "poster",
    "story",
    "wallpaper",
    "unknown",
]

ImageAction = Literal[
    "generate",
    "edit",
    "remove_object",
    "cleanup",
    "upscale",
    "avatar_crop",
    "logo_enhance",
    "background_replace",
    "background_remove",
    "variation",
]


@dataclass(frozen=True)
class ImagePlan:
    action: ImageAction
    kind: ImageKind
    aspect_ratio: str
    openai_size: str
    quality_notes: str
    safety_notes: str


def _text(value: str | None) -> str:
    return (value or "").strip()


def _lower(value: str | None) -> str:
    return _text(value).lower()


def _has_any(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)


def detect_image_kind(prompt: str | None) -> ImageKind:
    text = _lower(prompt)

    if _has_any(text, ["логотип", "лого", "logo", "brand mark", "иконка бренда"]):
        return "logo"
    if _has_any(text, ["аватар", "ава", "avatar", "profile picture", "юзерпик"]):
        return "avatar"
    if _has_any(text, ["интерфейс", "приложение", "app screen", "ui", "ux", "dashboard", "лендинг", "сайт"]):
        return "ui"
    if _has_any(text, ["мем", "meme", "смешная картинка"]):
        return "meme"
    if _has_any(text, ["постер", "афиша", "обложка", "cover", "poster", "баннер"]):
        return "poster"
    if _has_any(text, ["сторис", "story", "reels", "shorts", "тикток", "вертикальное видео"]):
        return "story"
    if _has_any(text, ["обои", "wallpaper", "фон", "background"]):
        return "wallpaper"
    if _has_any(text, ["товар", "product", "карточка товара", "маркетплейс"]):
        return "product"
    if _has_any(text, ["документ", "чек", "таблица", "скрин", "screenshot", "receipt", "document"]):
        return "document"
    if _has_any(text, ["фото", "реалист", "realistic", "photorealistic", "камера", "portrait"]):
        return "photo"
    if _has_any(text, ["арт", "иллюстрац", "рисунок", "anime", "comic", "painting", "art"]):
        return "art"

    return "unknown"


def detect_image_action(prompt: str | None, has_source_image: bool = False) -> ImageAction:
    text = _lower(prompt)

    if _has_any(text, ["убери", "удали", "remove", "erase", "без "]):
        return "remove_object"
    if _has_any(text, ["убери фон", "удали фон", "без фона", "remove background", "transparent background", "прозрачный фон"]):
        return "background_remove" if has_source_image else "generate"
    if _has_any(text, ["вариация", "вариант", "variation", "похожие варианты", "ещё варианты"]):
        return "variation" if has_source_image else "generate"
    if _has_any(text, ["фон", "background", "замени фон", "поменяй фон"]):
        return "background_replace" if has_source_image else "generate"
    if _has_any(text, ["почисти", "очисти", "убрать мусор", "cleanup", "clean up", "выровняй"]):
        return "cleanup"
    if _has_any(text, ["улучши качество", "upscale", "апскейл", "четче", "резче", "hd", "4k"]):
        return "upscale"
    if _has_any(text, ["квадратную аву", "аватар", "avatar", "крупным планом"]):
        return "avatar_crop" if has_source_image else "generate"
    if _has_any(text, ["логотип", "лого", "logo"]):
        return "logo_enhance" if has_source_image else "generate"

    return "edit" if has_source_image else "generate"


def infer_aspect_ratio(prompt: str | None, kind: ImageKind | None = None) -> str:
    text = _lower(prompt)
    kind = kind or detect_image_kind(prompt)

    # Explicit formats first.
    if _has_any(text, ["640x360", "16:9", "ютуб", "youtube", "баннер", "thumbnail", "превью", "обложка youtube"]):
        return "16:9"
    if _has_any(text, ["9:16", "сторис", "stories", "story", "reels", "рилс", "shorts", "тикток", "tiktok", "вертикаль", "вертикальная"]):
        return "9:16"
    if _has_any(text, ["4:3", "презентац", "presentation", "слайд"]):
        return "4:3"
    if _has_any(text, ["3:4", "портрет", "portrait", "вертикальный пост", "портретный пост"]):
        return "3:4"
    if _has_any(text, ["квадрат", "square", "1:1", "аватар", "ава"]):
        return "1:1"

    # Instagram is not always square. Default feed posts look better as portrait.
    if _has_any(text, ["instagram", "инстаграм", "инста", "insta"]):
        if _has_any(text, ["пост", "лента", "feed", "publication", "рекламный креатив"]):
            return "3:4"
        return "9:16"

    defaults = {
        "avatar": "1:1",
        "logo": "1:1",
        "ui": "9:16",
        "poster": "16:9",
        "story": "9:16",
        "wallpaper": "16:9",
        "product": "1:1",
        "meme": "1:1",
        "document": "3:4",
        "photo": "3:4",
        "art": "1:1",
        "unknown": "1:1",
    }
    return defaults.get(kind, "1:1")


def infer_openai_image_size(prompt: str | None) -> str:
    """Map flexible user formats to sizes supported by GPT Image.

    OpenAI image sizes are limited, so 3:4 / Instagram portrait uses the
    closest vertical HD canvas. The prompt itself tells the model to keep safe
    Instagram framing inside that canvas.
    """
    ratio = infer_aspect_ratio(prompt)
    if ratio == "16:9":
        return "1536x1024"
    if ratio in {"9:16", "3:4"}:
        return "1024x1536"
    return "1024x1024"


def infer_openai_image_quality(prompt: str | None, has_source_image: bool = False) -> str:
    """Balanced speed/quality.

    High quality is reserved for enhancement/upscale/logo/avatar/photo-edit
    requests. Regular random generations stay medium so GPT Image does not take
    several minutes for every casual prompt.
    """
    text = _lower(prompt)
    if has_source_image:
        return "high"
    if _has_any(text, ["hd", "4k", "качество", "улучши", "улучшить", "upscale", "апскейл", "логотип", "лого", "avatar", "аватар", "фото", "realistic", "реалист"]):
        return "high"
    return "medium"


def infer_gemini_aspect_ratio(prompt: str | None) -> str:
    ratio = infer_aspect_ratio(prompt)
    if ratio in {"1:1", "3:4", "4:3", "9:16", "16:9"}:
        return ratio
    return "1:1"


def build_image_plan(prompt: str | None, has_source_image: bool = False) -> ImagePlan:
    kind = detect_image_kind(prompt)
    action = detect_image_action(prompt, has_source_image=has_source_image)
    aspect_ratio = infer_aspect_ratio(prompt, kind)

    quality_by_kind = {
        "logo": "premium minimal vector-like mark, clean geometry, memorable silhouette, balanced negative space, app-icon ready, no random letters",
        "avatar": "centered subject, close-up composition, sharp focus, clean background, strong readable silhouette",
        "ui": "modern premium mobile interface, clean spacing, realistic app layout, readable structure without fake small text",
        "meme": "clear composition, expressive subject, simple background, no unreadable fake text unless requested",
        "poster": "cinematic composition, strong focal point, professional lighting, high contrast, clean typography only if requested",
        "story": "vertical composition, strong central subject, social-media ready framing, clean background",
        "wallpaper": "wide cinematic composition, depth, atmosphere, high detail, clean edges",
        "product": "commercial product shot, clean studio lighting, premium background, sharp details",
        "document": "clean readable layout, high contrast, document-like composition, no fake unreadable text unless requested",
        "photo": "photorealistic, natural lighting, realistic lens depth, detailed textures",
        "art": "high quality illustration, strong composition, coherent style, polished details",
        "unknown": "high quality, coherent composition, detailed subject, clean background, professional lighting",
    }

    safety_notes = (
        "Follow the user's subject literally. Do not add random captions, watermarks, logos, extra fingers, fake UI text, "
        "unreadable letters, duplicated objects or unrelated details. Preserve requested colors, composition, identity and brand words only when explicitly requested."
    )

    return ImagePlan(
        action=action,
        kind=kind,
        aspect_ratio=aspect_ratio,
        openai_size=infer_openai_image_size(prompt),
        quality_notes=quality_by_kind.get(kind, quality_by_kind["unknown"]),
        safety_notes=safety_notes,
    )


def build_image_generation_prompt(user_prompt: str | None, image_model: str = "image") -> str:
    original = _text(user_prompt) or "Create a high quality detailed image."
    plan = build_image_plan(original, has_source_image=False)

    return (
        f"User request: {original}\n\n"
        f"Image model: {image_model}\n"
        f"Detected type: {plan.kind}\n"
        f"Target aspect ratio: {plan.aspect_ratio}\n"
        "Use the requested aspect ratio deliberately. For Instagram/feed posts, keep the subject inside safe margins and avoid forced square composition unless requested.\n"
        f"Quality direction: {plan.quality_notes}\n"
        f"Rules: {plan.safety_notes}\n\n"
        "Create one polished final image. The result must match the user's request literally. Avoid visual clutter and random text."
    )[:3000]


def build_image_edit_prompt(user_prompt: str | None) -> str:
    original = _text(user_prompt) or "Improve this image while keeping the same subject."
    plan = build_image_plan(original, has_source_image=True)

    action_instructions = {
        "remove_object": "Remove only the requested object or distraction. Reconstruct the background naturally.",
        "background_replace": "Replace or improve the background as requested while keeping the main subject unchanged.",
        "background_remove": "Remove the background cleanly. Keep the main subject sharp, centered and natural. Use a transparent or clean plain background when supported.",
        "variation": "Create a polished variation of the original image while preserving the main subject, composition idea and recognizable identity.",
        "cleanup": "Clean visual noise, stray objects, bad edges, clutter, artifacts, and distracting background elements.",
        "upscale": "Improve clarity, sharpness, texture, edges and overall quality without changing the identity or composition.",
        "avatar_crop": "Create a clean square avatar-style composition with the subject larger and centered.",
        "logo_enhance": "Enhance the logo shape, symmetry, contrast, and premium look without adding unrelated symbols.",
        "edit": "Apply the requested edit while preserving the original subject and important details.",
        "generate": "Apply the requested visual change while preserving the source image context.",
    }

    return (
        f"Edit request: {original}\n\n"
        f"Detected action: {plan.action}\n"
        f"Detected type: {plan.kind}\n"
        f"Instruction: {action_instructions.get(plan.action, action_instructions['edit'])}\n"
        f"Quality direction: {plan.quality_notes}\n"
        "Preserve the original subject, identity, pose, perspective, aspect idea and important details unless the user explicitly asks to change them. "
        "For background removal, keep edges clean and avoid cutting off hair, hands, logos or object details. "
        "For upscale, improve quality without changing the picture content. "
        f"Rules: {plan.safety_notes}"
    )[:2500]


def image_result_caption(model_key: str, action: str = "generate") -> str:
    # Чистый caption без лишних иконок: в Telegram маленькие preview-иконки
    # рядом с подписью выглядят как визуальный мусор.
    return "Готово"


def image_filename(model_key: str, action: str = "generate") -> str:
    safe_model = re.sub(r"[^a-zA-Z0-9_-]+", "_", model_key or "image").strip("_") or "image"
    suffix = "edited" if action == "edit" else "generated"
    return f"{safe_model}_{suffix}.png"

