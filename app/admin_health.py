import os
from datetime import datetime

from app.ai.router import router_status


def _ok(value: bool) -> str:
    return "✅" if value else "❌"


def _enabled_env(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def _human_error(details: str | None) -> str:
    raw = (details or '').replace('\n', ' ').strip()
    low = raw.lower()
    if 'insufficient_quota' in low or 'exceeded your current quota' in low:
        return 'Закончился баланс или лимит OpenAI API. Нужно пополнить баланс в OpenAI Billing.'
    if '429' in low or 'rate limit' in low:
        return 'Лимит запросов API. Нужно подождать или проверить лимиты провайдера.'
    if 'timeout' in low or 'timed out' in low:
        return 'Таймаут: внешний AI-сервис слишком долго отвечал.'
    if 'loading_task' in low and 'not defined' in low:
        return 'Внутренняя ошибка обработки файла: loading_task не был создан.'
    if 'getupdates' in low or 'telegramconflicterror' in low:
        return 'Запущены два polling-экземпляра бота одновременно.'
    return raw[:220] if raw else 'без деталей'


def _human_event(event_type: str | None) -> str:
    return {
        'voice_error': 'ошибка голосового',
        'file_error': 'ошибка файла',
        'image_error': 'ошибка изображения',
        'vision_error': 'ошибка анализа фото',
        'ai_provider_error': 'ошибка AI-провайдера',
        'db_error': 'ошибка базы данных',
    }.get(event_type or '', event_type or '—')


async def build_admin_health_text(db_pool) -> str:
    """Admin health summary for Telegram /health command.

    Does not make paid external API calls; it checks configuration, DB and recent internal stats.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    webhook_mode = _enabled_env("WEBHOOK_MODE")
    redis_url = os.getenv("REDIS_URL")
    router = router_status("gpt", "text")
    router_providers = router.get("providers", {})
    router_order = ", ".join(router.get("order", [])) or "нет доступных провайдеров"

    db_ok = False
    total_users = 0
    events_24h = 0
    ai_24h = 0
    voice_24h = 0
    file_24h = 0
    image_24h = 0
    last_errors = []

    try:
        async with db_pool.acquire() as conn:
            db_ok = (await conn.fetchval("SELECT 1")) == 1
            total_users = await conn.fetchval("SELECT COUNT(*) FROM users") or 0
            events_24h = await conn.fetchval("""
                SELECT COUNT(*) FROM events
                WHERE created_at >= NOW() - INTERVAL '24 hours'
            """) or 0
            ai_24h = await conn.fetchval("""
                SELECT COUNT(*) FROM events
                WHERE event_type IN ('ai_message','ai_vision','ai_image','ai_image_edit','ai_file','ai_voice')
                  AND created_at >= NOW() - INTERVAL '24 hours'
            """) or 0
            voice_24h = await conn.fetchval("""
                SELECT COUNT(*) FROM events
                WHERE event_type='ai_voice' AND created_at >= NOW() - INTERVAL '24 hours'
            """) or 0
            file_24h = await conn.fetchval("""
                SELECT COUNT(*) FROM events
                WHERE event_type='ai_file' AND created_at >= NOW() - INTERVAL '24 hours'
            """) or 0
            image_24h = await conn.fetchval("""
                SELECT COUNT(*) FROM events
                WHERE event_type IN ('ai_image','ai_image_edit','ai_vision')
                  AND created_at >= NOW() - INTERVAL '24 hours'
            """) or 0
            last_errors = await conn.fetch("""
                SELECT event_type, details, created_at
                FROM events
                WHERE event_type ILIKE '%error%'
                ORDER BY created_at DESC
                LIMIT 5
            """)
    except Exception as exc:
        db_ok = False
        last_errors = [{"event_type": "db_error", "details": str(exc)[:160], "created_at": None}]

    error_text = "нет записанных ошибок"
    if last_errors:
        rows = []
        for row in last_errors:
            if isinstance(row, dict):
                event_type = row.get("event_type")
                details = row.get("details")
                created_at = row.get("created_at")
            else:
                event_type = row["event_type"]
                details = row["details"]
                created_at = row["created_at"]
            created = created_at.strftime("%d.%m %H:%M") if created_at else "сейчас"
            rows.append(f"• {created} — {_human_event(event_type)}: {_human_error(details)}")
        error_text = "\n".join(rows)

    return (
        "🩺 Здоровье VSotahBot\n\n"
        f"⏱ Проверка: {now}\n"
        f"🚀 Режим: {'webhook' if webhook_mode else 'polling'}\n\n"
        "🔌 Провайдеры:\n"
        f"• OpenAI: {_ok(bool(os.getenv('OPENAI_API_KEY')))}\n"
        f"• Claude: {_ok(bool(os.getenv('ANTHROPIC_API_KEY')))}\n"
        f"• Gemini: {_ok(bool(os.getenv('GOOGLE_API_KEY')))}\n"
        f"• DeepSeek: {_ok(bool(os.getenv('DEEPSEEK_API_KEY')))}\n"
        f"• Интернет-поиск: {_ok(bool(os.getenv('TAVILY_API_KEY')))}\n"
        f"• Redis: {_ok(bool(redis_url))}\n"
        f"• PostgreSQL: {_ok(db_ok)}\n\n"
        "📊 Активность за 24ч:\n"
        f"• всего пользователей: {total_users}\n"
        f"• всего событий: {events_24h}\n"
        f"• AI-запросы: {ai_24h}\n"
        f"• голосовых запросов: {voice_24h}\n"
        f"• файлов обработано: {file_24h}\n"
        f"• изображений/анализа фото: {image_24h}\n\n"
        f"⚠️ Последние ошибки:\n{error_text}"
    )
