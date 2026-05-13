import os
from io import BytesIO
from typing import Any

from openai import AsyncOpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
OPENAI_STT_LANGUAGE = os.getenv("OPENAI_STT_LANGUAGE", "").strip()
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
VOICE_TRANSCRIPT_LIMIT = int(os.getenv("VOICE_TRANSCRIPT_LIMIT", "5000"))
VOICE_TTS_TEXT_LIMIT = int(os.getenv("VOICE_TTS_TEXT_LIMIT", "1800"))

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


class VoiceProviderUnavailable(Exception):
    pass


def _safe_text(value: Any) -> str:
    return (value or "").strip()


def _clip_text(text: str, limit: int) -> str:
    text = _safe_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


async def transcribe_voice(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    """Speech-to-text for Telegram voice/audio files."""
    if not openai_client:
        raise VoiceProviderUnavailable("OPENAI_API_KEY is missing")
    if not audio_bytes:
        return ""

    audio_file = BytesIO(audio_bytes)
    audio_file.name = filename

    kwargs = {
        "model": OPENAI_STT_MODEL,
        "file": audio_file,
    }
    # Empty language means automatic language detection. Set OPENAI_STT_LANGUAGE=ru
    # in Railway if you want to force Russian-only recognition.
    if OPENAI_STT_LANGUAGE:
        kwargs["language"] = OPENAI_STT_LANGUAGE

    response = await openai_client.audio.transcriptions.create(**kwargs)

    text = _safe_text(getattr(response, "text", ""))
    return _clip_text(text, VOICE_TRANSCRIPT_LIMIT)


async def text_to_speech(text: str) -> bytes:
    """Optional AI voice reply. Disabled in bot_app unless VOICE_REPLY_ENABLED=true."""
    if not openai_client:
        raise VoiceProviderUnavailable("OPENAI_API_KEY is missing")

    speech_text = _clip_text(text, VOICE_TTS_TEXT_LIMIT)
    if not speech_text:
        return b""

    response = await openai_client.audio.speech.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=speech_text,
        response_format="mp3",
    )

    if hasattr(response, "aread"):
        return await response.aread()
    if hasattr(response, "read"):
        maybe = response.read()
        if hasattr(maybe, "__await__"):
            return await maybe
        return maybe
    if hasattr(response, "content"):
        return response.content

    return b""


def build_voice_user_message(transcript: str) -> str:
    transcript = _safe_text(transcript)
    return (
        "Пользователь отправил голосовое сообщение. "
        "Распознанный текст:\n\n"
        f"{transcript}"
    )


