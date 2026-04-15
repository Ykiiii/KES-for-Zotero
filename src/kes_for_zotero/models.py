from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ZoteroRelatedFile:
    path: Path
    kind: str
    size_bytes: int
    preview: str | None = None
    structured_fields: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ZoteroItem:
    item_key: str
    citation_key: str
    title: str
    year: str | None
    first_author: str | None
    item_dir: Path
    pdf_files: list[Path]
    related_files: list[ZoteroRelatedFile]


@dataclass(slots=True)
class ExtractedImage:
    path: Path
    relative_path: str
    width: int
    height: int
    context_excerpt: str = ""


@dataclass(slots=True)
class VisionAnalysis:
    keep: bool
    level: str
    figure_type: str
    title: str
    summary: str
    rationale: str
    raw_response: str = ""


@dataclass(slots=True)
class MarkerResult:
    pdf_path: Path
    markdown_path: Path
    markdown: str
    metadata: dict[str, Any]
    images: list[ExtractedImage] = field(default_factory=list)
    abstract: str | None = None
    conclusion: str | None = None


@dataclass(slots=True)
class ProcessedDocument:
    marker: MarkerResult
    level2_images: list[tuple[ExtractedImage, VisionAnalysis]] = field(default_factory=list)
    level3_images: list[tuple[ExtractedImage, VisionAnalysis]] = field(default_factory=list)