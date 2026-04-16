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
    model_cache_dir: Path = Path(".cache-marker/models")


@dataclass(slots=True)
class VisionSettings:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:11434"
    model: str = "gemma4"
    temperature: float = 0.1
    request_timeout_seconds: int = 180
    max_failures_per_pdf: int = 3
    max_analysis_seconds_per_pdf: int = 900
    include_level3: bool = False
    max_candidate_images: int = 24
    min_image_area: int = 50_000
    min_short_side: int = 160
    max_image_area: int | None = None
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 6.0
    deduplicate_images: bool = True
    drop_context_keywords: tuple[str, ...] = ()
    prefer_context_keywords: tuple[str, ...] = (
        "figure",
        "fig",
        "architecture",
        "pipeline",
        "framework",
        "result",
        "ablation",
        "comparison",
        "table",
        "图",
        "结果",
        "架构",
        "流程",
        "对比",
    )


@dataclass(slots=True)
class RunSettings:
    resume: bool = True
    show_progress: bool = True
    retry_attempts: int = 1
    force_reprocess: bool = False
    strict_fail: bool = False
    parallel_workers: int = 1
    suppress_internal_progress: bool = True
    sample_size: int | None = None
    gpu_mode: str = "single"
    gpu_devices: tuple[int, ...] = ()


@dataclass(slots=True)
class AppConfig:
    storage_root: Path
    output_root: Path
    marker: MarkerSettings = field(default_factory=MarkerSettings)
    vision: VisionSettings = field(default_factory=VisionSettings)
    run: RunSettings = field(default_factory=RunSettings)

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


def _coerce_optional_path(value: str | Path | None, base_dir: Path, fallback: Path) -> Path:
    path = _coerce_path(value, fallback)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def build_config(
    *,
    config_path: Path | None,
    storage_root: str | None,
    output_root: str | None,
    include_level3: bool | None,
    disable_vision: bool,
    item_model: str | None,
    enable_resume: bool | None = None,
    enable_progress: bool | None = None,
    retry_attempts: int | None = None,
    force_reprocess: bool | None = None,
    strict_fail: bool | None = None,
    parallel_workers: int | None = None,
    sample_size: int | None = None,
    gpu_mode: str | None = None,
    gpu_devices: tuple[int, ...] | None = None,
) -> AppConfig:
    file_config = _load_json_config(config_path) if config_path else {}
    config_base_dir = config_path.parent.resolve() if config_path else Path.cwd()

    marker_config = file_config.get("marker", {})
    vision_config = file_config.get("vision", {})
    run_config = file_config.get("run", {})

    marker = MarkerSettings(
        force_ocr=marker_config.get("force_ocr", True),
        use_llm=marker_config.get("use_llm", True),
        ollama_base_url=marker_config.get("ollama_base_url", "http://127.0.0.1:11434"),
        ollama_model=marker_config.get("ollama_model", item_model or "gemma4"),
        llm_service=marker_config.get("llm_service", "marker.services.ollama.OllamaService"),
        strip_existing_ocr=marker_config.get("strip_existing_ocr", False),
        page_range=marker_config.get("page_range"),
        model_cache_dir=_coerce_optional_path(
            marker_config.get("model_cache_dir"),
            config_base_dir,
            Path(".cache-marker/models"),
        ),
    )

    vision_enabled = vision_config.get("enabled", True)
    if disable_vision:
        vision_enabled = False

    vision = VisionSettings(
        enabled=vision_enabled,
        base_url=vision_config.get("base_url", marker.ollama_base_url),
        model=vision_config.get("model", item_model or marker.ollama_model),
        temperature=vision_config.get("temperature", 0.1),
        request_timeout_seconds=max(
            1,
            int(vision_config.get("request_timeout_seconds", 180)),
        ),
        max_failures_per_pdf=max(
            1,
            int(vision_config.get("max_failures_per_pdf", 3)),
        ),
        max_analysis_seconds_per_pdf=max(
            1,
            int(vision_config.get("max_analysis_seconds_per_pdf", 900)),
        ),
        include_level3=include_level3 if include_level3 is not None else vision_config.get("include_level3", False),
        max_candidate_images=vision_config.get("max_candidate_images", 24),
        min_image_area=vision_config.get("min_image_area", 50_000),
        min_short_side=vision_config.get("min_short_side", 160),
        max_image_area=vision_config.get("max_image_area"),
        min_aspect_ratio=float(vision_config.get("min_aspect_ratio", 0.2)),
        max_aspect_ratio=float(vision_config.get("max_aspect_ratio", 6.0)),
        deduplicate_images=bool(vision_config.get("deduplicate_images", True)),
        drop_context_keywords=tuple(vision_config.get("drop_context_keywords", [])),
        prefer_context_keywords=tuple(
            vision_config.get(
                "prefer_context_keywords",
                [
                    "figure",
                    "fig",
                    "architecture",
                    "pipeline",
                    "framework",
                    "result",
                    "ablation",
                    "comparison",
                    "table",
                    "图",
                    "结果",
                    "架构",
                    "流程",
                    "对比",
                ],
            )
        ),
    )

    run = RunSettings(
        resume=bool(enable_resume) if enable_resume is not None else bool(run_config.get("resume", True)),
        show_progress=bool(enable_progress)
        if enable_progress is not None
        else bool(run_config.get("show_progress", True)),
        retry_attempts=max(
            0,
            int(retry_attempts)
            if retry_attempts is not None
            else int(run_config.get("retry_attempts", 1)),
        ),
        force_reprocess=bool(force_reprocess)
        if force_reprocess is not None
        else bool(run_config.get("force_reprocess", False)),
        strict_fail=bool(strict_fail) if strict_fail is not None else bool(run_config.get("strict_fail", False)),
        parallel_workers=max(
            1,
            int(parallel_workers)
            if parallel_workers is not None
            else int(run_config.get("parallel_workers", 1)),
        ),
        suppress_internal_progress=bool(run_config.get("suppress_internal_progress", True)),
        sample_size=(
            max(1, int(sample_size))
            if sample_size is not None
            else (
                max(1, int(run_config["sample_size"]))
                if run_config.get("sample_size") is not None
                else None
            )
        ),
        gpu_mode=(gpu_mode or str(run_config.get("gpu_mode", "single")).strip().lower() or "single"),
        gpu_devices=(
            tuple(gpu_devices)
            if gpu_devices is not None
            else tuple(int(device) for device in run_config.get("gpu_devices", []) if isinstance(device, int))
        ),
    )

    if run.gpu_mode not in {"single", "round-robin"}:
        raise ValueError("Invalid run.gpu_mode. Supported values: single, round-robin")

    storage_root_value = storage_root or file_config.get("storage_root")
    if storage_root_value is None:
        raise ValueError("Missing storage_root. Provide --storage-root or set storage_root in config.")

    output_root_value = output_root or file_config.get("output_root")

    return AppConfig(
        storage_root=_coerce_optional_path(storage_root_value, config_base_dir, Path(".")),
        output_root=_coerce_optional_path(output_root_value, config_base_dir, Path("output")),
        marker=marker,
        vision=vision,
        run=run,
    )