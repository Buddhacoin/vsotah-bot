import re

def build_referral_link(bot_username: str, telegram_id: int) -> str:
    return f"https://t.me/{bot_username}?start=ref_{telegram_id}"

def parse_referral_code(text: str):
    if not text:
        return None

    match = re.search(r"ref_(\d+)", text)
    if not match:
        return None

    return int(match.group(1))
