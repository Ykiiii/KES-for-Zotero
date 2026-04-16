from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from kes_for_zotero.models import ZoteroItem, ZoteroRelatedFile

TEXT_PREVIEW_SUFFIXES = {
    ".json",
    ".txt",
    ".html",
    ".htm",
    ".xml",
    ".csv",
}

TEXT_PREVIEW_NAME_SUFFIXES = {
    ".zotero-ft-cache",
    ".zotero-ft-info",
}


def scan_storage(storage_root: Path, item_key: str | None = None) -> list[ZoteroItem]:
    if not storage_root.exists():
        raise FileNotFoundError(f"Storage root does not exist: {storage_root}")

    items: list[ZoteroItem] = []
    for child in sorted(storage_root.iterdir()):
        if not child.is_dir():
            continue
        if item_key and child.name != item_key:
            continue

        pdf_files = sorted(file for file in child.iterdir() if file.is_file() and file.suffix.lower() == ".pdf")
        related_files = [
            build_related_file(file)
            for file in sorted(child.iterdir())
            if file.is_file() and file.suffix.lower() != ".pdf"
        ]

        title, year, first_author = derive_bibliographic_metadata(related_files, pdf_files)
        citation_key = build_citation_key(first_author, year, title, child.name)

        if pdf_files or related_files:
            items.append(
                ZoteroItem(
                    item_key=child.name,
                    citation_key=citation_key,
                    title=title,
                    year=year,
                    first_author=first_author,
                    item_dir=child,
                    pdf_files=pdf_files,
                    related_files=related_files,
                )
            )

    return items


def build_related_file(path: Path) -> ZoteroRelatedFile:
    kind = classify_related_file(path)
    return ZoteroRelatedFile(
        path=path,
        kind=kind,
        size_bytes=path.stat().st_size,
        preview=read_text_preview(path),
        structured_fields=parse_structured_fields(path, kind),
    )


def classify_related_file(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".zotero-ft-info"):
        return "zotero-fulltext-info"
    if name.endswith(".zotero-ft-cache"):
        return "zotero-fulltext-cache"
    if path.suffix.lower() in {".html", ".htm"}:
        return "snapshot"
    if path.suffix.lower() in {".json", ".bib"}:
        return "metadata"
    return "attachment"


def read_text_preview(path: Path, limit: int = 1200) -> str | None:
    suffixes = "".join(path.suffixes).lower()
    name_lower = path.name.lower()
    is_special_zotero_text = any(name_lower.endswith(suffix) for suffix in TEXT_PREVIEW_NAME_SUFFIXES)

    if (
        not is_special_zotero_text
        and suffixes not in TEXT_PREVIEW_SUFFIXES
        and path.suffix.lower() not in TEXT_PREVIEW_SUFFIXES
    ):
        return None

    try:
        raw_text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None

    if not raw_text:
        return None

    if is_special_zotero_text:
        return truncate_text(raw_text, limit, preserve_lines=True)

    if suffixes.endswith(".json") or path.suffix.lower() == ".json":
        try:
            parsed = json.loads(raw_text)
            raw_text = json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass

    compact = " ".join(raw_text.split())
    return truncate_text(compact, limit, preserve_lines=False)


def truncate_text(text: str, limit: int, *, preserve_lines: bool) -> str:
    normalized = text.strip()
    if not normalized:
        return ""

    if not preserve_lines:
        normalized = " ".join(normalized.split())

    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def parse_structured_fields(path: Path, kind: str) -> dict[str, str]:
    if kind == "zotero-fulltext-info":
        return parse_zotero_ft_info(path)
    if kind == "zotero-fulltext-cache":
        return parse_zotero_ft_cache(path)
    return {}


def parse_zotero_ft_info(path: Path) -> dict[str, str]:
    try:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}

    fields: dict[str, str] = {}
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            fields[key] = value
    return fields


def parse_zotero_ft_cache(path: Path) -> dict[str, str]:
    try:
        raw_text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return {}
    if not raw_text:
        return {}

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    normalized = "\n".join(lines)

    title = ""
    abstract = ""

    if lines:
        title = lines[0]
        if title.lower() == "r e s e a r c h open access" and len(lines) > 1:
            title = lines[1]

    abstract_match = re.search(
        r"(?is)\babstract\b\s*[:\n]\s*(.+?)(?:\n\s*\n|\n\s*background\b|\n\s*methods\b)",
        normalized,
    )
    if abstract_match:
        abstract = " ".join(abstract_match.group(1).split())

    result = {
        "line_count": str(len(lines)),
        "char_count": str(len(raw_text)),
    }
    if title:
        result["detected_title"] = title
    if abstract:
        result["detected_abstract"] = truncate_text(abstract, 800, preserve_lines=False)
    return result


def derive_bibliographic_metadata(related_files: list[ZoteroRelatedFile], pdf_files: list[Path]) -> tuple[str, str | None, str | None]:
    info = next((rf for rf in related_files if rf.kind == "zotero-fulltext-info"), None)
    cache = next((rf for rf in related_files if rf.kind == "zotero-fulltext-cache"), None)

    title = ""
    year: str | None = None
    first_author: str | None = None

    if info is not None:
        title = info.structured_fields.get("Title", "")
        first_author = extract_first_author(info.structured_fields.get("Author"))
        year = extract_year(info.structured_fields.get("Subject")) or extract_year(info.structured_fields.get("CreationDate"))

    if not title and cache is not None:
        title = cache.structured_fields.get("detected_title", "")

    if pdf_files:
        pdf_title, pdf_year, pdf_first_author = parse_pdf_filename_metadata(pdf_files[0].stem)
        if not title:
            title = pdf_title
        if not year:
            year = pdf_year
        if not first_author:
            first_author = pdf_first_author

    return title or "Untitled", year, first_author


def extract_first_author(author_field: str | None) -> str | None:
    if not author_field:
        return None
    author = author_field.split(",", 1)[0].strip()
    author = author.split(" and ", 1)[0].strip()
    tokens = [token for token in re.split(r"\s+", author) if token]
    if not tokens:
        return None
    surname = tokens[-1]
    surname = normalize_key_token(surname)
    return surname or None


def extract_year(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", text)
    if not match:
        return None
    return match.group(0)


def build_citation_key(first_author: str | None, year: str | None, title: str, fallback: str) -> str:
    # Better BibTeX-like pattern: [auth][year][shorttitle].
    author_part = normalize_key_token(first_author or "anon")
    year_part = year or "nodate"
    shorttitle_part = extract_short_title_token(title)
    key = f"{author_part}{year_part}{shorttitle_part}"
    key = normalize_key_token(key)
    return key or normalize_key_token(fallback)


def extract_short_title_token(title: str) -> str:
    stopwords = {
        "a",
        "an",
        "the",
        "of",
        "on",
        "for",
        "toward",
        "towards",
        "in",
        "and",
        "to",
    }
    normalized = unicodedata.normalize("NFKD", title)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    tokens = re.findall(r"[A-Za-z0-9]+", normalized.lower())
    for token in tokens:
        if token not in stopwords:
            return token
    return "item"


def parse_pdf_filename_metadata(stem: str) -> tuple[str, str | None, str | None]:
    parts = [part.strip() for part in stem.split(" - ")]
    if len(parts) >= 3:
        author_part = parts[0]
        year_part = extract_year(parts[1])
        title_part = " - ".join(parts[2:]).strip()
        first_author = extract_first_author_from_filename(author_part)
        if title_part:
            return title_part, year_part, first_author
    return stem, extract_year(stem), extract_first_author_from_filename(stem)


def extract_first_author_from_filename(text: str) -> str | None:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.replace("et al.", " ")
    normalized = normalized.replace("et al", " ")
    normalized = normalized.replace("等", " ")
    normalized = normalized.replace("和", " ")
    normalized = normalized.replace("&", " ")
    normalized = normalized.replace(" and ", " ")
    tokens = re.findall(r"[A-Za-z]+", normalized)
    if not tokens:
        return None
    return normalize_key_token(tokens[0]) or None


def normalize_key_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "", normalized)
    return normalized