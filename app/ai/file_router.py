import asyncio
import base64
import os
from io import BytesIO
from typing import Any

from openai import AsyncOpenAI

from app.ai.prompts import clean_ai_answer, system_prompt
from app.ai.personalities import get_personality
from app.ai.router import ai_router
from app.ai.vision_router import build_pdf_vision_prompt


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.4-mini")
FILE_TEXT_LIMIT = int(os.getenv("FILE_TEXT_LIMIT", "22000"))
FILE_MAX_PAGES = int(os.getenv("FILE_MAX_PAGES", "35"))
FILE_XLSX_MAX_ROWS = int(os.getenv("FILE_XLSX_MAX_ROWS", "120"))
FILE_XLSX_MAX_SHEETS = int(os.getenv("FILE_XLSX_MAX_SHEETS", "6"))
PDF_VISION_MAX_PAGES = int(os.getenv("PDF_VISION_MAX_PAGES", "4"))
TEXT_HISTORY_LIMIT = int(os.getenv("TEXT_HISTORY_LIMIT", "6"))
VISION_HISTORY_LIMIT = int(os.getenv("VISION_HISTORY_LIMIT", "2"))
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "900"))
AI_TIMEOUT_SECONDS = int(os.getenv("AI_TIMEOUT_SECONDS", "75"))

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


SUPPORTED_FILE_TYPES = {
    "txt", "md", "csv", "log", "json", "html", "htm",
    "pdf", "docx", "xlsx", "xlsm",
}


def _extension(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1] if "." in filename else ""


def _limit_text(text: str) -> str:
    return (text or "")[:FILE_TEXT_LIMIT]


def _decode_text_file(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return _limit_text(file_bytes.decode(encoding, errors="replace"))
        except Exception:
            continue
    return _limit_text(file_bytes.decode("utf-8", errors="replace"))


def _extract_pdf_text(file_bytes: bytes) -> str:
    pages: list[str] = []

    # 1) pypdf — fast for regular PDFs with a text layer.
    try:
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(file_bytes))
        for index, page in enumerate(reader.pages[:FILE_MAX_PAGES], start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"--- Страница {index} ---\n{text.strip()}")
            if len("\n\n".join(pages)) >= FILE_TEXT_LIMIT:
                return _limit_text("\n\n".join(pages))
    except Exception as e:
        print(f"PDF PYPDF READ ERROR: {e}")

    # 2) pdfplumber — often better for tables and invoices.
    if not "\n".join(pages).strip():
        try:
            import pdfplumber
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for index, page in enumerate(pdf.pages[:FILE_MAX_PAGES], start=1):
                    text = page.extract_text() or ""
                    tables = []
                    try:
                        for table in page.extract_tables() or []:
                            rows = [" | ".join(str(cell or "") for cell in row) for row in table[:30]]
                            if rows:
                                tables.append("\n".join(rows))
                    except Exception:
                        pass
                    page_text = "\n".join(part for part in [text.strip(), *tables] if part.strip())
                    if page_text:
                        pages.append(f"--- Страница {index} ---\n{page_text}")
                    if len("\n\n".join(pages)) >= FILE_TEXT_LIMIT:
                        return _limit_text("\n\n".join(pages))
        except Exception as e:
            print(f"PDF PDFPLUMBER READ ERROR: {e}")

    # 3) PyMuPDF fallback.
    if not "\n".join(pages).strip():
        try:
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for index, page in enumerate(doc[:FILE_MAX_PAGES], start=1):
                text = page.get_text("text") or ""
                if text.strip():
                    pages.append(f"--- Страница {index} ---\n{text.strip()}")
                if len("\n\n".join(pages)) >= FILE_TEXT_LIMIT:
                    break
            doc.close()
        except Exception as e:
            print(f"PDF FITZ TEXT READ ERROR: {e}")

    return _limit_text("\n\n".join(pages))


def _extract_docx_text(file_bytes: bytes) -> str:
    try:
        from docx import Document
        doc = Document(BytesIO(file_bytes))
        parts: list[str] = []

        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                parts.append(text)

        for table_index, table in enumerate(doc.tables, start=1):
            parts.append(f"\n--- Таблица {table_index} ---")
            for row in table.rows[:80]:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
                if len("\n".join(parts)) >= FILE_TEXT_LIMIT:
                    return _limit_text("\n".join(parts))

        return _limit_text("\n".join(parts))
    except Exception as e:
        raise ValueError(f"DOCX_READ_ERROR: {e}")


def _extract_xlsx_text(file_bytes: bytes) -> str:
    try:
        from openpyxl import load_workbook
        wb = load_workbook(BytesIO(file_bytes), data_only=True, read_only=True)
        lines: list[str] = []

        for ws in wb.worksheets[:FILE_XLSX_MAX_SHEETS]:
            lines.append(f"--- Лист: {ws.title} ---")
            for row in ws.iter_rows(max_row=FILE_XLSX_MAX_ROWS, values_only=True):
                values = [str(v) if v is not None else "" for v in row]
                if any(v.strip() for v in values):
                    lines.append(" | ".join(values))
                if len("\n".join(lines)) >= FILE_TEXT_LIMIT:
                    return _limit_text("\n".join(lines))

        return _limit_text("\n".join(lines))
    except Exception as e:
        raise ValueError(f"XLSX_READ_ERROR: {e}")


def extract_document_text(filename: str, file_bytes: bytes) -> str:
    """Extracts readable text/data from user files for Files Pipeline 1.0."""
    ext = _extension(filename)

    if ext not in SUPPORTED_FILE_TYPES:
        raise ValueError("UNSUPPORTED_FILE_TYPE")

    if ext in {"txt", "md", "csv", "log", "json", "html", "htm"}:
        return _decode_text_file(file_bytes)

    if ext == "pdf":
        return _extract_pdf_text(file_bytes)

    if ext == "docx":
        return _extract_docx_text(file_bytes)

    if ext in {"xlsx", "xlsm"}:
        return _extract_xlsx_text(file_bytes)

    raise ValueError("UNSUPPORTED_FILE_TYPE")


def detect_file_kind(filename: str, extracted_text: str = "") -> str:
    ext = _extension(filename)
    lower = f"{filename}\n{extracted_text[:2000]}".lower()

    if ext in {"xlsx", "xlsm", "csv"}:
        return "spreadsheet"
    if ext == "pdf" and not extracted_text.strip():
        return "scanned_pdf"
    if any(word in lower for word in ("договор", "contract", "agreement", "акт", "счет", "invoice")):
        return "business_document"
    if any(word in lower for word in ("диплом", "курсов", "реферат", "dissertation", "thesis")):
        return "study_document"
    if ext == "docx":
        return "word_document"
    if ext == "pdf":
        return "pdf_document"
    return "text_document"


def build_file_analysis_prompt(question: str, filename: str, extracted_text: str) -> str:
    file_kind = detect_file_kind(filename, extracted_text)
    question = question.strip() or "Проанализируй файл и выдели главное."

    return (
        "Ты — premium файловый аналитик VSotah AI. "
        "Отвечай дорого, профессионально, понятно и структурированно.\n\n"
        f"Тип файла: {file_kind}\n"
        f"Имя файла: {filename}\n"
        f"Вопрос пользователя: {question}\n\n"
        "Правила ответа:\n"
        "1. Не выдумывай данные, которых нет в файле.\n"
        "2. Сначала дай краткое summary.\n"
        "3. Затем выдели главные выводы списком.\n"
        "4. Для таблиц ищи закономерности, аномалии, рост, падение и важные цифры.\n"
        "5. Для договоров выделяй риски, сроки, обязательства и потенциально опасные пункты.\n"
        "6. Для учебных материалов объясняй простым языком.\n"
        "7. Используй аккуратные markdown-блоки и списки.\n"
        "8. В конце всегда добавляй блок «Что можно сделать дальше».\n\n"
        "Содержимое файла:\n"
        f"{_limit_text(extracted_text)}"
    )


def build_file_status_text(filename: str, stage: str) -> str:
    ext = _extension(filename).upper() or "FILE"

    if stage == "reading":
        return (
            "📄 Обрабатываю файл...\n"
            f"⚡ Читаю {ext}: {filename}"
        )

    if stage == "analyzing":
        return (
            "🧠 VSotah AI анализирует файл...\n"
            "📊 Извлекаю ключевые данные...\n"
            "✨ Формирую AI-выводы..."
        )

    if stage == "scanned_pdf":
        return (
            "📄 PDF похож на скан.\n"
            "🖼 Анализирую страницы как изображения..."
        )

    return "📎 Обрабатываю файл..."


def render_pdf_pages_to_images(file_bytes: bytes, max_pages: int = PDF_VISION_MAX_PAGES) -> list[bytes]:
    try:
        import fitz
    except Exception:
        return []

    images: list[bytes] = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_index in range(min(len(doc), max_pages)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            images.append(pix.tobytes("jpeg"))
        doc.close()
    except Exception as e:
        print(f"PDF RENDER ERROR: {e}")
        return []

    return images


async def analyze_pdf_images_with_openai(
    question: str,
    filename: str,
    file_bytes: bytes,
    history: list[dict],
    selected_model: str = "gpt",
) -> str:
    """Vision fallback for scanned PDFs and image-only documents."""
    page_images = await asyncio.to_thread(render_pdf_pages_to_images, file_bytes, PDF_VISION_MAX_PAGES)
    if not page_images or not openai_client:
        return ""

    question = question.strip() or "Проанализируй PDF, прочитай видимый текст и выдели главное."
    content: list[dict[str, Any]] = [
        {"type": "text", "text": build_pdf_vision_prompt(filename, question)}
    ]

    for img in page_images:
        image_b64 = base64.b64encode(img).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "low",
                },
            }
        )

    system_text = system_prompt(get_personality(selected_model))
    openai_messages: list[dict[str, Any]] = [{"role": "system", "content": system_text}]
    for msg in history[-VISION_HISTORY_LIMIT:]:
        if msg.get("role") in {"user", "assistant"} and msg.get("content"):
            openai_messages.append({"role": msg["role"], "content": msg["content"]})
    openai_messages.append({"role": "user", "content": content})

    try:
        response = await asyncio.wait_for(
            openai_client.chat.completions.create(
                model=OPENAI_VISION_MODEL,
                messages=openai_messages,
                temperature=0.4,
                max_completion_tokens=VISION_MAX_TOKENS,
            ),
            timeout=AI_TIMEOUT_SECONDS,
        )
        return clean_ai_answer(response.choices[0].message.content or "")
    except Exception as e:
        print(f"PDF VISION ERROR: {str(e).replace(chr(10), ' ')[:1200]}")
        return ""


async def file_router(
    selected_model: str,
    question: str,
    filename: str,
    extracted_text: str,
    history: list[dict],
) -> str:
    """Routes extracted file content through the user's selected text AI model."""
    prompt = build_file_analysis_prompt(question, filename, extracted_text)
    messages = [*history[-TEXT_HISTORY_LIMIT:], {"role": "user", "content": prompt}]
    return await ai_router(selected_model, messages)



