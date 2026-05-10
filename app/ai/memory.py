def build_memory(messages, limit=12):
    """
    Собирает короткую память диалога
    """

    history = []

    for msg in messages[-limit:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if not content:
            continue

        history.append({
            "role": role,
            "content": content[:4000]
        })

    return history
