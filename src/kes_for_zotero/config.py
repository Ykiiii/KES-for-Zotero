from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MarkerSettings:
    force_ocr: bool = True
    use_llm: bool = True
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma4"
    llm_service: str = "marker.services.ollama.OllamaService"
    strip_existing_ocr: bool = False
    page_range: str | None = None


@dataclass(slots=True)
class VisionSettings:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:11434"
    model: str = "gemma4"
    temperature: float = 0.1
    include_level3: bool = False
    max_candidate_images: int = 24
    min_image_area: int = 50_000
    min_short_side: int = 160


@dataclass(slots=True)
class AppConfig:
    storage_root: Path
    output_root: Path
    marker: MarkerSettings = field(default_factory=MarkerSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "manifest.json"


def _load_json_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_path(value: str | Path | None, fallback: Path | None = None) -> Path:
    if value is None:
        if fallback is None:
            raise ValueError("Missing required path configuration.")
        return fallback
    return Path(value).expanduser()


def build_config(
    *,
    config_path: Path | None,
    storage_root: str | None,
    output_root: str | None,
    include_level3: bool | None,
    disable_vision: bool,
    item_model: str | None,
) -> AppConfig:
    file_config = _load_json_config(config_path) if config_path else {}

    marker_config = file_config.get("marker", {})
    vision_config = file_config.get("vision", {})

    marker = MarkerSettings(
        force_ocr=marker_config.get("force_ocr", True),
        use_llm=marker_config.get("use_llm", True),
        ollama_base_url=marker_config.get("ollama_base_url", "http://127.0.0.1:11434"),
        ollama_model=marker_config.get("ollama_model", item_model or "gemma4"),
        llm_service=marker_config.get("llm_service", "marker.services.ollama.OllamaService"),
        strip_existing_ocr=marker_config.get("strip_existing_ocr", False),
        page_range=marker_config.get("page_range"),
    )

    vision_enabled = vision_config.get("enabled", True)
    if disable_vision:
        vision_enabled = False

    vision = VisionSettings(
        enabled=vision_enabled,
        base_url=vision_config.get("base_url", marker.ollama_base_url),
        model=vision_config.get("model", item_model or marker.ollama_model),
        temperature=vision_config.get("temperature", 0.1),
        include_level3=include_level3 if include_level3 is not None else vision_config.get("include_level3", False),
        max_candidate_images=vision_config.get("max_candidate_images", 24),
        min_image_area=vision_config.get("min_image_area", 50_000),
        min_short_side=vision_config.get("min_short_side", 160),
    )

    return AppConfig(
        storage_root=_coerce_path(storage_root or file_config.get("storage_root")),
        output_root=_coerce_path(output_root or file_config.get("output_root"), Path("output")),
        marker=marker,
        vision=vision,
    )