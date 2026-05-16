import os
from datetime import datetime

from app.ai.router import router_status


def _ok(value: bool) -> str:
    return "✅" if value else "❌"


def _enabled_env(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


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
            created = created_at.strftime("%d.%m %H:%M") if created_at else "now"
            rows.append(f"• {created} — {event_type}: {(details or '')[:120]}")
        error_text = "\n".join(rows)

    return (
        "🩺 Health VSotahBot\n\n"
        f"⏱ Проверка: {now}\n"
        f"🚀 Режим: {'webhook' if webhook_mode else 'polling'}\n\n"
        "🔌 Провайдеры:\n"
        f"• OpenAI: {_ok(bool(os.getenv('OPENAI_API_KEY')))}\n"
        f"• Claude: {_ok(bool(os.getenv('ANTHROPIC_API_KEY')))}\n"
        f"• Gemini: {_ok(bool(os.getenv('GOOGLE_API_KEY')))}\n"
        f"• DeepSeek: {_ok(bool(os.getenv('DEEPSEEK_API_KEY')))}\n"
        f"• Tavily Internet: {_ok(bool(os.getenv('TAVILY_API_KEY')))}\n"
        f"• Redis: {_ok(bool(redis_url))}\n"
        f"• PostgreSQL: {_ok(db_ok)}\n\n"
        "📊 Активность за 24ч:\n"
        f"• users total: {total_users}\n"
        f"• events: {events_24h}\n"
        f"• AI-запросы: {ai_24h}\n"
        f"• voice: {voice_24h}\n"
        f"• files: {file_24h}\n"
        f"• image/vision: {image_24h}\n\n"
        f"⚠️ Последние ошибки:\n{error_text}"
    )
