from __future__ import annotations

import json
from pathlib import Path

from kes_for_zotero.models import ZoteroItem, ZoteroRelatedFile

TEXT_PREVIEW_SUFFIXES = {
    ".json",
    ".txt",
    ".html",
    ".htm",
    ".xml",
    ".csv",
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

        if pdf_files or related_files:
            items.append(
                ZoteroItem(
                    item_key=child.name,
                    item_dir=child,
                    pdf_files=pdf_files,
                    related_files=related_files,
                )
            )

    return items


def build_related_file(path: Path) -> ZoteroRelatedFile:
    return ZoteroRelatedFile(
        path=path,
        kind=classify_related_file(path),
        size_bytes=path.stat().st_size,
        preview=read_text_preview(path),
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
    if suffixes not in TEXT_PREVIEW_SUFFIXES and path.suffix.lower() not in TEXT_PREVIEW_SUFFIXES:
        return None

    try:
        raw_text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None

    if not raw_text:
        return None

    if suffixes.endswith(".json") or path.suffix.lower() == ".json":
        try:
            parsed = json.loads(raw_text)
            raw_text = json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass

    compact = " ".join(raw_text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."