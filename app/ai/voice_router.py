import asyncio
import mimetypes
import os
from io import BytesIO
from typing import Any

from openai import AsyncOpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
OPENAI_STT_LANGUAGE = os.getenv("OPENAI_STT_LANGUAGE", "").strip()
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "nova")
OPENAI_TTS_FORMAT = os.getenv("OPENAI_TTS_FORMAT", "opus")
VOICE_TRANSCRIPT_LIMIT = int(os.getenv("VOICE_TRANSCRIPT_LIMIT", "5000"))
VOICE_STT_TIMEOUT_SECONDS = int(os.getenv("VOICE_STT_TIMEOUT_SECONDS", "45"))
# Voice replies must be short: Telegram users already received the full text answer.
# Shorter TTS is much faster on Railway and uploads back to Telegram quicker.
VOICE_TTS_TEXT_LIMIT = int(os.getenv("VOICE_TTS_TEXT_LIMIT", "700"))

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


def _clean_tts_text(text: str) -> str:
    """Make AI answer shorter and easier for speech synthesis."""
    text = _safe_text(text)
    replacements = {
        "**": "",
        "__": "",
        "```": "",
        "###": "",
        "##": "",
        "#": "",
        "•": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _safe_audio_filename(filename: str = "voice.ogg") -> str:
    filename = (filename or "voice.ogg").strip()
    if "." not in filename:
        filename += ".ogg"
    return filename


def _audio_mime_type(filename: str) -> str:
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "audio/ogg"


def _stt_models_to_try() -> list[str]:
    # OPENAI_STT_MODEL is configurable, then we keep two safe fallbacks.
    # This is intentionally independent from the chat model GPT-5.
    candidates = [
        OPENAI_STT_MODEL,
        "gpt-4o-mini-transcribe",
        "whisper-1",
    ]
    result = []
    for model in candidates:
        model = (model or "").strip()
        if model and model not in result:
            result.append(model)
    return result


async def transcribe_voice(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    """Speech-to-text for Telegram voice/audio files.

    Telegram voice notes are usually OGG/OPUS. The OpenAI SDK is more reliable
    when we pass the file as an explicit (filename, bytes, mime_type) tuple
    instead of a reused BytesIO object.
    """
    if not openai_client:
        raise VoiceProviderUnavailable("OPENAI_API_KEY is missing")
    if not audio_bytes:
        return ""

    safe_filename = _safe_audio_filename(filename)
    mime_type = _audio_mime_type(safe_filename)
    last_error: Exception | None = None

    for model in _stt_models_to_try():
        kwargs = {
            "model": model,
            "file": (safe_filename, audio_bytes, mime_type),
        }
        # Empty language means automatic language detection. Set OPENAI_STT_LANGUAGE=ru
        # in Railway if you want to force Russian-only recognition.
        if OPENAI_STT_LANGUAGE:
            kwargs["language"] = OPENAI_STT_LANGUAGE

        try:
            response = await asyncio.wait_for(
                openai_client.audio.transcriptions.create(**kwargs),
                timeout=VOICE_STT_TIMEOUT_SECONDS,
            )
            if isinstance(response, str):
                text = _safe_text(response)
            elif isinstance(response, dict):
                text = _safe_text(response.get("text", ""))
            else:
                text = _safe_text(getattr(response, "text", ""))
            if text:
                return _clip_text(text, VOICE_TRANSCRIPT_LIMIT)
        except Exception as error:
            last_error = error
            print(f"VOICE STT MODEL ERROR | {model}: {type(error).__name__}: {str(error)[:500]}")
            continue

    if last_error:
        raise last_error
    return ""


async def text_to_speech(text: str) -> bytes:
    """Optional AI voice reply. Disabled in bot_app unless VOICE_REPLY_ENABLED=true."""
    if not openai_client:
        raise VoiceProviderUnavailable("OPENAI_API_KEY is missing")

    speech_text = _clip_text(_clean_tts_text(text), VOICE_TTS_TEXT_LIMIT)
    if not speech_text:
        return b""

    response = await openai_client.audio.speech.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=speech_text,
        response_format=OPENAI_TTS_FORMAT,
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

