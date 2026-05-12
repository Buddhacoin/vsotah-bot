import os
from typing import Literal

IMAGE_PROVIDER = os.getenv('IMAGE_PROVIDER', 'openai')
GPT_IMAGE_MODEL = os.getenv('GPT_IMAGE_MODEL', 'gpt-image-1')
NANO_BANANA_MODEL = os.getenv('NANO_BANANA_MODEL', 'imagen-4.0-generate-001')

ImageType = Literal['avatar','photo','art','logo','ui','meme','document','unknown']


def detect_image_type(prompt: str) -> ImageType:
    text = (prompt or '').lower()
    if any(x in text for x in ['logo','логотип']):
        return 'logo'
    if any(x in text for x in ['avatar','ава','аватар']):
        return 'avatar'
    if any(x in text for x in ['ui','interface','интерфейс']):
        return 'ui'
    return 'unknown'


def enhance_image_prompt(prompt: str) -> str:
    image_type = detect_image_type(prompt)

    styles = {
        'logo': 'minimalistic, premium, cinematic lighting, clean geometry, ultra detailed',
        'avatar': 'sharp focus, centered composition, detailed face, dramatic lighting',
        'ui': 'modern ui, glassmorphism, premium mobile app interface',
        'unknown': 'high quality, cinematic, ultra detailed'
    }

    return f'{prompt}\n\nStyle: {styles.get(image_type, styles["unknown"])}'

