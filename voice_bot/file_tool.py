# File-reading tool: read text-ish files and return their content as a string.

from __future__ import annotations

import base64
import logging
import zipfile
from pathlib import Path
from typing import Optional

from . import config as cfg

logger = logging.getLogger("voice_bot.file_tool")

# Optional document libraries (graceful fallback if missing)
try:
    import docx
except ImportError:
    docx = None  # type: ignore
try:
    import openpyxl
except ImportError:
    openpyxl = None  # type: ignore
try:
    import pptx
except ImportError:
    pptx = None  # type: ignore
try:
    import fitz  # pymupdf
except ImportError:
    fitz = None  # type: ignore

# Extension -> reader function
_READERS: dict[str, callable] = {}


def _ensure_readers() -> None:
    if _READERS:
        return

    def _read_text(path: Path, encoding: str = "utf-8") -> str:
        p = Path(path)
        try:
            return p.read_text(encoding=encoding)
        except UnicodeDecodeError:
            return p.read_text(encoding="latin-1")

    def _read_json(path: Path) -> str:
        import json
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            return f"[JSON PARSE ERROR: {e}]\n\nRAW:\n{p.read_text(encoding='utf-8', errors='replace')}"
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _read_csv(path: Path) -> str:
        import csv
        p = Path(path)
        rows = []
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
        lines = [",".join(row) for row in rows]
        return "\n".join(lines)

    def _read_docx(path: Path) -> str:
        if docx is None:
            raise RuntimeError("python-docx not installed")
        d = docx.Document(str(path))
        parts = []
        for p in d.paragraphs:
            if p.text:
                parts.append(p.text)
        for table in d.tables:
            for row in table.rows:
                cells = [c.text for c in row.cells]
                parts.append(" | ".join(cells))
        return "\n".join(parts)

    def _read_xlsx(path: Path) -> str:
        if openpyxl is None:
            raise RuntimeError("openpyxl not installed")
        wb = openpyxl.load_workbook(str(path), data_only=True)
        parts = []
        for ws in wb.worksheets:
            parts.append(f"=== Sheet: {ws.title} ===")
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(cells):
                    parts.append("\t".join(cells))
        return "\n".join(parts)

    def _read_pptx(path: Path) -> str:
        if pptx is None:
            raise RuntimeError("python-pptx not installed")
        prs = pptx.Presentation(str(path))
        parts = []
        for i, slide in enumerate(prs.slides, 1):
            parts.append(f"--- Slide {i} ---")
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        if para.text:
                            parts.append(para.text)
                if shape.has_table:
                    table = shape.table
                    for row in table.rows:
                        cells = [c.text for c in row.cells]
                        parts.append(" | ".join(cells))
        return "\n".join(parts)

    def _read_pdf(path: Path) -> str:
        if fitz is None:
            raise RuntimeError("pymupdf (fitz) not installed")
        doc = fitz.open(str(path))
        parts = []
        for page in doc:
            parts.append(page.get_text())
        return "\n".join(parts)

    def _read_ini(path: Path) -> str:
        return _read_text(path)

    def _read_yaml(path: Path) -> str:
        try:
            import yaml
            p = Path(path)
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            import json
            return json.dumps(data, indent=2, ensure_ascii=False)
        except ImportError:
            return _read_text(path)

    def _read_toml(path: Path) -> str:
        try:
            import tomli
            p = Path(path)
            data = tomli.load(p.read_bytes())
            import json
            return json.dumps(data, indent=2, ensure_ascii=False)
        except ImportError:
            return _read_text(path)

    _READERS[".txt"] = _read_text
    _READERS[".md"] = _read_text
    _READERS[".py"] = _read_text
    _READERS[".js"] = _read_text
    _READERS[".ts"] = _read_text
    _READERS[".jsx"] = _read_text
    _READERS[".tsx"] = _read_text
    _READERS[".html"] = _read_text
    _READERS[".htm"] = _read_text
    _READERS[".css"] = _read_text
    _READERS[".json"] = _read_json
    _READERS[".csv"] = _read_csv
    _READERS[".xml"] = _read_text
    _READERS[".yaml"] = _read_yaml
    _READERS[".yml"] = _read_yaml
    _READERS[".toml"] = _read_toml
    _READERS[".log"] = _read_text
    _READERS[".sh"] = _read_text
    _READERS[".bash"] = _read_text
    _READERS[".bat"] = _read_text
    _READERS[".ps1"] = _read_text
    _READERS[".cfg"] = _read_ini
    _READERS[".ini"] = _read_ini
    _READERS[".docx"] = _read_docx
    _READERS[".xlsx"] = _read_xlsx
    _READERS[".pptx"] = _read_pptx
    _READERS[".pdf"] = _read_pdf


def read_file(path: Path, config: Optional[cfg.Config] = None) -> str:
    """Read a file and return its content as a string.

    - Supports a wide range of text/document formats.
    - Respects the configured max read size (skips huge files).
    - Returns an error string on failure (caller should check for error markers).
    """
    _ensure_readers()
    config = config or cfg  # module-level fallback

    p = Path(path)
    if not p.exists():
        return f"[ERROR: file not found: {p}]"
    if not p.is_file():
        return f"[ERROR: not a file: {p}]"

    ext = p.suffix.lower()
    size = p.stat().st_size
    max_bytes = (config.FILES.max_read_bytes if hasattr(config, "FILES") else cfg.FILES.max_read_bytes)

    if size > max_bytes:
        return (f"[ERROR: file too large to read ({size} bytes > {max_bytes} bytes). "
                f"Path: {p}]")

    reader = _READERS.get(ext)
    if reader is None:
        # Fallback: try plain text
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"[ERROR: cannot read file type {ext or 'none'}: {e}]"

    try:
        content = reader(p)
    except RuntimeError as e:
        return f"[ERROR: missing dependency to read {ext}: {e}]"
    except Exception as e:
        return f"[ERROR: failed to read {p}: {e}]"

    # Truncate overly long content to keep LLM context manageable
    max_content_chars = max_bytes  # rough budget
    if len(content) > max_content_chars:
        content = (content[:max_content_chars] +
                   f"\n\n... [TRUNCATED: showing first {max_content_chars} chars of {len(content)} total] ...")

    return content


def extract_paths(text: str, base_dir: Optional[Path] = None) -> list[Path]:
    """Find absolute or relative file paths mentioned in text.

    Heuristic: look for paths that look like filesystem paths (contain a separator
    or start with a drive letter on Windows). Returns paths that exist on disk.
    """
    import re
    base_dir = base_dir or Path.cwd()

    # Match Windows or Unix paths
    candidates: list[str] = []

    # Absolute Windows paths like C:\...\... or C:/...\...
    candidates.extend(re.findall(r"[A-Za-z]:[\\\/][^\s\"'<>|;]+", text))
    # Absolute Unix paths
    candidates.extend(re.findall(r"(?:^|\s)(/[^\s\"'<>|;]+)", text))
    # Relative paths (word chars with a dot extension)
    candidates.extend(re.findall(r"[^\s\"'<>|;]+\.[a-zA-Z0-9]{1,6}", text))

    found: list[Path] = []
    seen: set[str] = set()
    for cand in candidates:
        cand = cand.strip('\'"')
        if not cand:
            continue
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)

        p: Path
        if re.match(r"^[A-Za-z]:", cand):
            p = Path(cand)
        elif cand.startswith("/"):
            p = Path(cand)
        else:
            p = (base_dir / cand).resolve()

        if p.exists() and p.is_file():
            found.append(p)
    return found
