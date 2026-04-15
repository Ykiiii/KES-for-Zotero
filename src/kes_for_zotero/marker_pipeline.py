from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Any

from kes_for_zotero.config import MarkerSettings, VisionSettings
from kes_for_zotero.models import ExtractedImage, MarkerResult

ABSTRACT_ALIASES = {"abstract", "摘要"}
CONCLUSION_KEYWORDS = {"conclusion", "conclusions", "结论", "summary and conclusion"}
_CONVERTER_INIT_LOCK = threading.Lock()


class MarkerExtractor:
    def __init__(self, settings: MarkerSettings) -> None:
        self.settings = settings
        self._converter: Any | None = None

    def extract(self, pdf_path: Path, item_output_dir: Path) -> MarkerResult:
        item_output_dir.mkdir(parents=True, exist_ok=True)
        marker_dir = item_output_dir / "marker"
        marker_dir.mkdir(parents=True, exist_ok=True)

        rendered = self._get_converter()(str(pdf_path))

        markdown = getattr(rendered, "markdown", "") or ""
        metadata = getattr(rendered, "metadata", {}) or {}
        images = getattr(rendered, "images", {}) or {}

        markdown_path = marker_dir / f"{pdf_path.stem}.marker.md"
        markdown_path.write_text(markdown, encoding="utf-8")

        asset_root = item_output_dir / "assets" / pdf_path.stem
        extracted_images = self._save_images(images, markdown, asset_root, item_output_dir)

        return MarkerResult(
            pdf_path=pdf_path,
            markdown_path=markdown_path,
            markdown=markdown,
            metadata=metadata,
            images=extracted_images,
            abstract=extract_named_section(markdown, ABSTRACT_ALIASES),
            conclusion=extract_conclusion(markdown),
        )

    def candidate_images(self, result: MarkerResult, vision: VisionSettings) -> list[ExtractedImage]:
        filtered: list[tuple[int, int, ExtractedImage]] = []
        seen_hashes: set[str] = set()

        for image in result.images:
            width = max(0, image.width)
            height = max(0, image.height)
            area = width * height
            if area < vision.min_image_area:
                continue
            if vision.max_image_area is not None and area > vision.max_image_area:
                continue
            if min(width, height) < vision.min_short_side:
                continue

            aspect_ratio = _safe_aspect_ratio(width, height)
            if aspect_ratio < vision.min_aspect_ratio or aspect_ratio > vision.max_aspect_ratio:
                continue

            normalized_context = image.context_excerpt.lower()
            if _contains_any(normalized_context, vision.drop_context_keywords):
                continue

            if vision.deduplicate_images:
                digest = _image_digest(image.path)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)

            priority = 1 if _contains_any(normalized_context, vision.prefer_context_keywords) else 0
            filtered.append((priority, area, image))

        filtered.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in filtered[: vision.max_candidate_images]]

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter

        with _CONVERTER_INIT_LOCK:
            if self._converter is not None:
                return self._converter

            self._configure_cache_environment()

            try:
                from marker.config.parser import ConfigParser
                from marker.converters.pdf import PdfConverter
                from marker.models import create_model_dict
            except ImportError as exc:
                raise RuntimeError(
                    "Failed to import Marker runtime dependencies. "
                    "Please verify marker-pdf, torch, torchvision, and transformers compatibility. "
                    f"Original error: {exc}"
                ) from exc

            config: dict[str, Any] = {
                "output_format": "markdown",
                "force_ocr": self.settings.force_ocr,
                "use_llm": self.settings.use_llm,
                "strip_existing_ocr": self.settings.strip_existing_ocr,
                "ollama_base_url": self.settings.ollama_base_url,
                "ollama_model": self.settings.ollama_model,
                "llm_service": self.settings.llm_service,
            }
            if self.settings.page_range:
                config["page_range"] = self.settings.page_range

            config_parser = ConfigParser(config)
            try:
                self._converter = PdfConverter(
                    config=config_parser.generate_config_dict(),
                    artifact_dict=create_model_dict(),
                    processor_list=config_parser.get_processors(),
                    renderer=config_parser.get_renderer(),
                    llm_service=config_parser.get_llm_service(),
                )
            except NotImplementedError as exc:
                message = str(exc)
                raise RuntimeError(
                    "Marker model initialization failed while moving model to target device. "
                    "This often happens with concurrent heavy model loads on multi-GPU hosts. "
                    "Try reducing --parallel-workers to match --gpu-devices count (or set to 1), "
                    "then rerun. "
                    f"Original error: {message}"
                ) from exc

            return self._converter

    def _configure_cache_environment(self) -> None:
        model_cache_dir = self.settings.model_cache_dir.resolve()
        cache_root = model_cache_dir.parent
        huggingface_root = cache_root / "huggingface"

        model_cache_dir.mkdir(parents=True, exist_ok=True)
        huggingface_root.mkdir(parents=True, exist_ok=True)
        (huggingface_root / "hub").mkdir(parents=True, exist_ok=True)
        (cache_root / "torch").mkdir(parents=True, exist_ok=True)

        os.environ["MODEL_CACHE_DIR"] = str(model_cache_dir)
        os.environ.setdefault("HF_HOME", str(huggingface_root))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(huggingface_root / "hub"))
        os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        # On Windows + MKL, limiting thread count avoids known KMeans memory leak in marker pipeline.
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    def _save_images(
        self,
        images: dict[str, Any],
        markdown: str,
        asset_root: Path,
        item_output_dir: Path,
    ) -> list[ExtractedImage]:
        asset_root.mkdir(parents=True, exist_ok=True)
        extracted: list[ExtractedImage] = []

        for raw_name, image_obj in images.items():
            safe_name = sanitize_asset_name(raw_name)
            output_path = asset_root / safe_name
            save_image_object(output_path, image_obj)
            width, height = inspect_image_size(output_path)
            relative_path = output_path.relative_to(item_output_dir).as_posix()
            context_excerpt = find_image_context(markdown, raw_name, safe_name)
            extracted.append(
                ExtractedImage(
                    path=output_path,
                    relative_path=relative_path,
                    width=width,
                    height=height,
                    context_excerpt=context_excerpt,
                )
            )

        return extracted


def sanitize_asset_name(raw_name: str) -> str:
    candidate = Path(raw_name).name or raw_name
    if "." not in candidate:
        candidate = candidate + ".png"
    return re.sub(r"[^A-Za-z0-9._-]", "_", candidate)


def save_image_object(output_path: Path, image_obj: Any) -> None:
    if hasattr(image_obj, "save"):
        image_obj.save(output_path)
        return
    if isinstance(image_obj, bytes):
        output_path.write_bytes(image_obj)
        return
    if isinstance(image_obj, str):
        try:
            output_path.write_bytes(base64.b64decode(image_obj, validate=True))
            return
        except (ValueError, binascii.Error):
            output_path.write_text(image_obj, encoding="utf-8")
            return
    if hasattr(image_obj, "getvalue"):
        output_path.write_bytes(image_obj.getvalue())
        return
    raise TypeError(f"Unsupported image payload type: {type(image_obj)!r}")


def inspect_image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError:
        return 0, 0

    try:
        with Image.open(image_path) as image:
            return image.size
    except OSError:
        return 0, 0


def extract_named_section(markdown: str, aliases: set[str]) -> str | None:
    headings = list(iter_headings(markdown))
    if not headings:
        return None

    lines = markdown.splitlines()
    for index, level, title in headings:
        normalized = normalize_heading(title)
        if normalized in aliases:
            section_lines: list[str] = []
            for next_index in range(index + 1, len(lines)):
                next_line = lines[next_index]
                if next_line.startswith("#"):
                    match = re.match(r"^(#+)\s+(.+)$", next_line)
                    if match and len(match.group(1)) <= level:
                        break
                section_lines.append(next_line)
            content = "\n".join(section_lines).strip()
            return content or None
    return None


def extract_conclusion(markdown: str) -> str | None:
    headings = list(iter_headings(markdown))
    if not headings:
        return None

    lines = markdown.splitlines()
    for index, level, title in headings:
        normalized = normalize_heading(title)
        if any(keyword in normalized for keyword in CONCLUSION_KEYWORDS):
            section_lines: list[str] = []
            for next_index in range(index + 1, len(lines)):
                next_line = lines[next_index]
                if next_line.startswith("#"):
                    match = re.match(r"^(#+)\s+(.+)$", next_line)
                    if match and len(match.group(1)) <= level:
                        break
                section_lines.append(next_line)
            content = "\n".join(section_lines).strip()
            return content or None
    return None


def iter_headings(markdown: str):
    for index, line in enumerate(markdown.splitlines()):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            yield index, len(match.group(1)), match.group(2)


def normalize_heading(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", "", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def find_image_context(markdown: str, raw_name: str, safe_name: str, window: int = 2) -> str:
    lines = markdown.splitlines()
    probes = {raw_name, Path(raw_name).name, safe_name, Path(safe_name).stem}
    for index, line in enumerate(lines):
        if any(probe and probe in line for probe in probes):
            start = max(0, index - window)
            end = min(len(lines), index + window + 1)
            return "\n".join(lines[start:end]).strip()
    return ""


def _safe_aspect_ratio(width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    return max(width / height, height / width)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    if not text or not keywords:
        return False
    return any(keyword and keyword.lower() in text for keyword in keywords)


def _image_digest(path: Path) -> str:
    hasher = hashlib.sha1()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError:
        return path.as_posix()
    return hasher.hexdigest()