from typing import Optional


def build_vision_prompt(user_text: str, image_context: Optional[str] = None) -> str:
    prompt = (
        'Ты Vision AI внутри VSotahBot. '
        'Анализируй изображения максимально точно и полезно.\n\n'
        f'Запрос пользователя: {user_text}'
    )

    if image_context:
        prompt += f'\n\nДополнительный контекст: {image_context}'

    return prompt
