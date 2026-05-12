import re
from urllib.parse import quote


def build_referral_link(bot_username: str, telegram_id: int) -> str:
    return f"https://t.me/{bot_username}?start=ref_{telegram_id}"


def parse_referral_code(text: str):
    if not text:
        return None

    match = re.search(r"ref_(\d+)", text)
    if not match:
        return None

    return int(match.group(1))


def build_invite_text(referral_link: str) -> str:
    return (
        "🚀 Я пользуюсь VSotah AI — тут ChatGPT, Claude, Gemini, генерация картинок, "
        "анализ фото, голосовые и работа с файлами в одном Telegram-боте.\n\n"
        "Заходи по моей ссылке, тестируй бесплатно и выбирай нужный AI:\n"
        f"{referral_link}"
    )


def build_telegram_share_url(referral_link: str) -> str:
    text = build_invite_text(referral_link)
    return f"https://t.me/share/url?url={quote(referral_link)}&text={quote(text)}"

