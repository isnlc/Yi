"""Parse uploaded documents into labeled text sections for memory import."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_EXTRACTED_CHARS = 80_000
MAX_SECTIONS = 40
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".log",
    ".pdf", ".docx", ".xlsx", ".pptx",
    *IMAGE_EXTENSIONS,
}
LEGACY_EXTENSIONS = {".doc", ".xls", ".ppt"}
OCR_SCRIPT = Path(__file__).with_name("windows_ocr.ps1")


class DocumentParseError(ValueError):
    pass


@dataclass
class ParsedDocument:
    filename: str
    kind: str
    sections: list[tuple[str, str]]
    warning: str = ""

    @property
    def char_count(self) -> int:
        return sum(len(text) for _label, text in self.sections)


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentParseError("无法识别文本文件编码。")


def _clean(text: str) -> str:
    lines = [line.strip() for line in text.replace("\x00", "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _normalize_ocr_text(text: str) -> str:
    """Remove artificial spaces inserted between adjacent Chinese characters."""
    return re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", text)


def _limit_sections(sections: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
    output: list[tuple[str, str]] = []
    total = 0
    truncated = False
    for label, raw_text in sections:
        text = _clean(raw_text)
        if not text:
            continue
        if len(output) >= MAX_SECTIONS or total >= MAX_EXTRACTED_CHARS:
            truncated = True
            break
        remaining = MAX_EXTRACTED_CHARS - total
        if len(text) > remaining:
            text = text[:remaining]
            truncated = True
        output.append((label, text))
        total += len(text)
    warning = "文档内容较长，已截取前 8 万字用于记忆。" if truncated else ""
    return output, warning


def _decode_process_output(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def ocr_path(path: Path, max_pages: int = 20) -> list[tuple[str, str]]:
    """Use the Windows built-in OCR engine for an image or rendered PDF pages."""
    if os.name != "nt":
        raise DocumentParseError("当前系统不支持 Windows OCR。")
    if not OCR_SCRIPT.exists():
        raise DocumentParseError("OCR 组件缺失。")

    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", str(OCR_SCRIPT),
        "-Path", str(path.resolve()),
        "-MaxPages", str(max(1, max_pages)),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise DocumentParseError("OCR 处理超时，请缩小文件后重试。") from exc
    except OSError as exc:
        raise DocumentParseError(f"OCR 启动失败：{exc}") from exc

    stdout = _decode_process_output(result.stdout).strip().lstrip("\ufeff")
    stderr = _decode_process_output(result.stderr).strip()
    if result.returncode != 0:
        detail = stderr or stdout or "未知错误"
        raise DocumentParseError(f"OCR 处理失败：{detail}")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise DocumentParseError("OCR 返回内容无法解析。") from exc

    sections = []
    for item in payload.get("sections", []):
        label = str(item.get("label", "")).strip() or "图片"
        text = _clean(_normalize_ocr_text(str(item.get("text", ""))))
        if text:
            sections.append((label, text))
    return sections


def _ocr_bytes(filename: str, data: bytes, max_pages: int = 20) -> list[tuple[str, str]]:
    suffix = Path(filename).suffix.lower()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        return ocr_path(temporary_path, max_pages=max_pages)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_plain(ext: str, data: bytes) -> list[tuple[str, str]]:
    text = _decode_text(data)
    if ext == ".json":
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
    elif ext == ".csv":
        try:
            rows = list(csv.reader(io.StringIO(text)))
            text = "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
        except csv.Error:
            pass
    return [("全文", text)]


def _parse_pdf(filename: str, data: bytes) -> tuple[list[tuple[str, str]], bool]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        sections = [
            (f"第 {index + 1} 页", page.extract_text() or "")
            for index, page in enumerate(reader.pages)
        ]
    except Exception as exc:
        raise DocumentParseError(f"PDF 解析失败：{exc}") from exc

    if any(_clean(text) for _label, text in sections):
        return sections, False

    ocr_sections = _ocr_bytes(filename, data, max_pages=min(len(sections) or 1, MAX_SECTIONS))
    if not ocr_sections:
        raise DocumentParseError("PDF 中没有识别到可用文字。")
    return ocr_sections, True


def _parse_docx(data: bytes) -> list[tuple[str, str]]:
    try:
        from docx import Document
        doc = Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentParseError(f"Word 解析失败：{exc}") from exc
    lines = [p.text for p in doc.paragraphs if p.text.strip()]
    for table_index, table in enumerate(doc.tables, 1):
        lines.append(f"[表格 {table_index}]")
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    return [("正文", "\n".join(lines))]


def _parse_xlsx(data: bytes) -> list[tuple[str, str]]:
    try:
        from openpyxl import load_workbook
        book = load_workbook(io.BytesIO(data), read_only=True, data_only=False)
    except Exception as exc:
        raise DocumentParseError(f"Excel 解析失败：{exc}") from exc
    sections = []
    try:
        for sheet in book.worksheets:
            lines = []
            for row_index, row in enumerate(sheet.iter_rows(values_only=True), 1):
                if row_index > 1000:
                    lines.append("……工作表行数过多，后续内容已省略")
                    break
                values = ["" if value is None else str(value) for value in row[:80]]
                while values and not values[-1]:
                    values.pop()
                if any(values):
                    lines.append(" | ".join(values))
            sections.append((f"工作表：{sheet.title}", "\n".join(lines)))
    finally:
        book.close()
    return sections


def _parse_pptx(data: bytes) -> list[tuple[str, str]]:
    try:
        from pptx import Presentation
        presentation = Presentation(io.BytesIO(data))
    except Exception as exc:
        raise DocumentParseError(f"PowerPoint 解析失败：{exc}") from exc
    sections = []
    for slide_index, slide in enumerate(presentation.slides, 1):
        lines = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                text = shape.text.strip()
                if text:
                    lines.append(text)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    lines.append(" | ".join(cell.text.strip() for cell in row.cells))
        try:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                lines.append("[演讲者备注]\n" + notes)
        except Exception:
            pass
        sections.append((f"第 {slide_index} 页", "\n".join(lines)))
    return sections


def parse_document(filename: str, data: bytes) -> ParsedDocument:
    safe_name = Path(filename or "uploaded").name
    ext = Path(safe_name).suffix.lower()
    if len(data) > MAX_FILE_BYTES:
        raise DocumentParseError("文件超过 50MB，请压缩或拆分后上传。")
    if ext in LEGACY_EXTENSIONS:
        modern = {".doc": ".docx", ".xls": ".xlsx", ".ppt": ".pptx"}[ext]
        raise DocumentParseError(f"旧版 {ext} 暂不直接解析，请另存为 {modern} 后上传。")
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError("不支持该文件格式。")

    warning = ""
    if ext in {".txt", ".md", ".csv", ".json", ".log"}:
        sections = _parse_plain(ext, data)
        kind = "文本"
    elif ext == ".pdf":
        sections, used_ocr = _parse_pdf(safe_name, data)
        kind = "扫描 PDF（OCR）" if used_ocr else "PDF"
        if used_ocr:
            warning = "扫描 PDF 已通过系统 OCR 识别，结果可能受清晰度和版式影响。"
    elif ext == ".docx":
        sections = _parse_docx(data)
        kind = "Word"
    elif ext == ".xlsx":
        sections = _parse_xlsx(data)
        kind = "Excel"
    elif ext == ".pptx":
        sections = _parse_pptx(data)
        kind = "PowerPoint"
    else:
        sections = _ocr_bytes(safe_name, data, max_pages=1)
        kind = "图片（OCR）"
        warning = "图片文字已通过系统 OCR 识别，不包含物体、场景等视觉理解。"

    sections, limit_warning = _limit_sections(sections)
    warning = " ".join(item for item in (warning, limit_warning) if item)
    if not sections:
        if ext in IMAGE_EXTENSIONS:
            raise DocumentParseError("图片中没有识别到清晰文字。")
        raise DocumentParseError("文件中没有提取到可用文字。")
    return ParsedDocument(filename=safe_name, kind=kind, sections=sections, warning=warning)
