def build_memory(messages: list[dict], limit: int = 12) -> list[dict]:
    """
    Собирает короткую память диалога для быстрой и стабильной работы.
    Оставляет последние сообщения, убирает пустые и слишком длинные куски.
    """
    if not messages:
        return []

    history = []
    for msg in messages[-limit:]:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip()

        if role not in {"user", "assistant", "system"}:
            continue
        if not content:
            continue

        history.append({
            "role": role,
            "content": content[:4000],
        })

    return history
