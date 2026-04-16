from __future__ import annotations

import contextlib
import io
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kes_for_zotero.config import AppConfig
from kes_for_zotero.marker_pipeline import MarkerExtractor
from kes_for_zotero.markdown_writer import render_catalog_entry, render_catalog_index, render_item_markdown
from kes_for_zotero.models import MarkerResult, ProcessedDocument, ZoteroItem
from kes_for_zotero.vision_llm import OllamaVisionClient
from kes_for_zotero.zotero_storage import scan_storage

LOGGER = logging.getLogger(__name__)

PAPPER_DIRNAME = "papper"
INDEX_DIRNAME = "index"


@dataclass(slots=True)
class _ProgressHandle:
    enabled: bool
    _bar: Any | None = None
    _heartbeat_thread: threading.Thread | None = None
    _heartbeat_stop: threading.Event | None = None

    def update(self, n: int = 1) -> None:
        if self._bar is not None:
            self._bar.update(n)

    def set_postfix_text(self, text: str) -> None:
        if self._bar is None:
            return
        try:
            self._bar.set_postfix_str(text)
            self._bar.refresh()
        except Exception:
            return

    def start_heartbeat(self, label: str) -> None:
        if self._bar is None or self._heartbeat_thread is not None:
            return
        stop_event = threading.Event()
        self._heartbeat_stop = stop_event
        started_at = time.monotonic()
        spinner = ["|", "/", "-", "\\"]

        def _runner() -> None:
            tick = 0
            while not stop_event.wait(1.0):
                elapsed = int(time.monotonic() - started_at)
                symbol = spinner[tick % len(spinner)]
                tick += 1
                self.set_postfix_text(f"{label} {symbol} {elapsed}s")

        self._heartbeat_thread = threading.Thread(target=_runner, daemon=True)
        self._heartbeat_thread.start()

    def close(self) -> None:
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2)
            self._heartbeat_thread = None
        if self._bar is not None:
            self._bar.close()


def run_pipeline(config: AppConfig, item_key: str | None = None) -> dict:
    config.output_root.mkdir(parents=True, exist_ok=True)
    papper_root = config.output_root / PAPPER_DIRNAME
    index_root = config.output_root / INDEX_DIRNAME
    papper_root.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)

    items = scan_storage(config.storage_root, item_key)
    items = _sample_items(items, config.run.sample_size)

    manifest = _load_manifest(config.manifest_path) if config.run.resume else {"items": []}
    item_records = _item_record_map(manifest)
    output_names = _assign_output_dir_names(items)
    worker_count = _resolve_worker_count(config, len(items))

    progress = _create_progress(len(items), config.run.show_progress)
    progress.start_heartbeat("papers")

    any_failure = False
    catalog_rows: list[dict[str, str]] = []

    def submit_payload(item_index: int, item: ZoteroItem) -> tuple[ZoteroItem, str, dict[str, Any]]:
        previous_item_record = item_records.get(item.item_key)
        output_dir_name = output_names[item.item_key]
        gpu_id = _select_gpu_id(config, item_index)
        item_record = _process_single_item(
            item=item,
            output_dir_name=output_dir_name,
            config=config,
            papper_root=papper_root,
            previous_item_record=previous_item_record,
            gpu_id=gpu_id,
        )
        return item, output_dir_name, item_record

    try:
        if worker_count > 1 and len(items) > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(submit_payload, item_index, item): item
                    for item_index, item in enumerate(items)
                }
                for future in as_completed(future_map):
                    item = future_map[future]
                    output_dir_name = output_names[item.item_key]
                    try:
                        item, output_dir_name, item_record = future.result()
                    except Exception as exc:
                        LOGGER.exception("Failed to process paper %s (%s)", item.citation_key, item.item_key)
                        item_record = _build_item_failure_record(
                            item=item,
                            output_dir_name=output_dir_name,
                            previous_item_record=item_records.get(item.item_key),
                            gpu_id=None,
                            error=exc,
                        )
                    any_failure = _finalize_item_record(
                        item=item,
                        output_dir_name=output_dir_name,
                        item_record=item_record,
                        manifest=manifest,
                        index_root=index_root,
                        catalog_rows=catalog_rows,
                        any_failure=any_failure,
                    )
                    progress.update(1)
                    _write_manifest(config.manifest_path, manifest)
                    _write_checkpoint_outputs(config.output_root, items, manifest)
        else:
            for item_index, item in enumerate(items):
                output_dir_name = output_names[item.item_key]
                try:
                    item, output_dir_name, item_record = submit_payload(item_index, item)
                except Exception as exc:
                    LOGGER.exception("Failed to process paper %s (%s)", item.citation_key, item.item_key)
                    item_record = _build_item_failure_record(
                        item=item,
                        output_dir_name=output_dir_name,
                        previous_item_record=item_records.get(item.item_key),
                        gpu_id=None,
                        error=exc,
                    )
                any_failure = _finalize_item_record(
                    item=item,
                    output_dir_name=output_dir_name,
                    item_record=item_record,
                    manifest=manifest,
                    index_root=index_root,
                    catalog_rows=catalog_rows,
                    any_failure=any_failure,
                )
                progress.update(1)
                _write_manifest(config.manifest_path, manifest)
                _write_checkpoint_outputs(config.output_root, items, manifest)
    finally:
        progress.close()

    _write_catalog_index(index_root, catalog_rows)
    _write_unfinished_units(config.output_root, items, manifest)
    _write_run_stats(config.output_root, items, manifest)

    if config.run.strict_fail and any_failure:
        raise RuntimeError("Pipeline finished with failures under strict_fail mode.")

    return manifest


def _process_single_item(
    *,
    item: ZoteroItem,
    output_dir_name: str,
    config: AppConfig,
    papper_root: Path,
    previous_item_record: dict[str, Any] | None,
    gpu_id: int | None,
) -> dict[str, Any]:
    LOGGER.info("Processing paper %s (%s)", item.citation_key, item.item_key)
    _activate_gpu(gpu_id)
    item_output_dir = papper_root / output_dir_name
    item_output_dir.mkdir(parents=True, exist_ok=True)

    marker = MarkerExtractor(config.marker)
    vision = OllamaVisionClient(config.vision) if config.vision.enabled else None

    item_record = previous_item_record or {"item_key": item.item_key, "pdfs": [], "status": "ok"}
    pdf_record_map = {
        entry.get("file"): entry
        for entry in item_record.get("pdfs", [])
        if isinstance(entry, dict) and entry.get("file")
    }
    item_record.update(
        {
            "item_key": item.item_key,
            "citation_key": item.citation_key,
            "title": item.title,
            "year": item.year,
            "output_dir": output_dir_name,
            "gpu_id": gpu_id,
            "pdfs": [],
            "status": "ok",
            "pdf_status": "present" if item.pdf_files else "missing",
        }
    )

    processed_documents: list[ProcessedDocument] = []
    item_progress = _create_item_progress(
        total=len(item.pdf_files),
        enabled=(
            config.run.show_progress
            and config.run.parallel_workers == 1
            and len(item.pdf_files) > 0
        ),
        citation_key=item.citation_key,
    )
    item_progress.start_heartbeat("pdf")

    try:
        for pdf_path in item.pdf_files:
            item_progress.set_postfix_text(pdf_path.name)
            existing_pdf_record = pdf_record_map.get(pdf_path.name)
            disk_completed = _is_pdf_completed_on_disk(pdf_path, item_output_dir)
            should_skip = (
                config.run.resume
                and not config.run.force_reprocess
                and (
                    (existing_pdf_record is not None and existing_pdf_record.get("status") == "ok")
                    or disk_completed
                )
            )

            if should_skip:
                LOGGER.info("Resuming: skip completed PDF %s for %s", pdf_path.name, item.citation_key)
                if existing_pdf_record is not None and existing_pdf_record.get("status") == "ok":
                    item_record["pdfs"].append(existing_pdf_record)
                else:
                    item_record["pdfs"].append(
                        {
                            "file": pdf_path.name,
                            "status": "ok",
                            "level2_images": 0,
                            "level3_images": 0,
                            "resume_source": "disk",
                        }
                    )
                cached_doc = _load_cached_document(pdf_path, item_output_dir)
                if cached_doc is not None:
                    processed_documents.append(cached_doc)
                item_progress.update(1)
                continue

            pdf_record = _process_single_pdf(
                pdf_path=pdf_path,
                item_output_dir=item_output_dir,
                marker=marker,
                vision=vision,
                config=config,
                processed_documents=processed_documents,
            )
            item_record["pdfs"].append(pdf_record)
            if pdf_record.get("status") != "ok":
                item_record["status"] = "partial-failure"
            item_progress.update(1)
    finally:
        item_progress.close()

    if not item.pdf_files:
        item_record["status"] = "indexed-only"

    markdown = render_item_markdown(item, processed_documents, item_output_dir)
    _write_item_indexes(item_output_dir, markdown, output_dir_name)
    return item_record


def _finalize_item_record(
    *,
    item: ZoteroItem,
    output_dir_name: str,
    item_record: dict[str, Any],
    manifest: dict[str, Any],
    index_root: Path,
    catalog_rows: list[dict[str, str]],
    any_failure: bool,
) -> bool:
    _upsert_item_record(manifest, item_record)
    item_index_path = index_root / f"{output_dir_name}.md"
    pdf_status = _display_pdf_status(item_record.get("pdf_status"))
    item_index_path.write_text(
        render_catalog_entry(item, Path(output_dir_name), pdf_status=pdf_status),
        encoding="utf-8",
    )
    catalog_rows.append(
        {
            "citation_key": item.citation_key,
            "title": item.title,
            "year": item.year or "",
            "pdf_status": pdf_status,
            "entry_path": f"./{output_dir_name}.md",
            "paper_index_path": f"../{PAPPER_DIRNAME}/{output_dir_name}/{output_dir_name}.index.md",
        }
    )
    if item_record.get("status") == "partial-failure":
        return True
    return any_failure


def _process_single_pdf(
    *,
    pdf_path: Path,
    item_output_dir: Path,
    marker: MarkerExtractor,
    vision: OllamaVisionClient | None,
    config: AppConfig,
    processed_documents: list[ProcessedDocument],
) -> dict[str, Any]:
    attempts = config.run.retry_attempts + 1

    def _extract() -> ProcessedDocument:
        marker_result = _extract_marker_result(
            pdf_path=pdf_path,
            item_output_dir=item_output_dir,
            marker=marker,
            suppress_internal_progress=config.run.suppress_internal_progress,
        )
        processed = ProcessedDocument(marker=marker_result)
        if vision is not None:
            vision_failures = 0
            started_at = time.monotonic()
            for image in marker.candidate_images(marker_result, config.vision):
                elapsed = int(time.monotonic() - started_at)
                if elapsed >= config.vision.max_analysis_seconds_per_pdf:
                    LOGGER.warning(
                        "Vision analysis budget exceeded for %s (%ss), skip remaining images.",
                        pdf_path.name,
                        config.vision.max_analysis_seconds_per_pdf,
                    )
                    break
                try:
                    analysis = _run_with_retries(
                        lambda: vision.analyze_image(image, pdf_path.name),
                        attempts=attempts,
                        action=f"vision analysis for {pdf_path.name}:{image.path.name}",
                    )
                except Exception as exc:
                    vision_failures += 1
                    LOGGER.warning(
                        "Skip image %s for %s due to vision error (%s/%s): %s",
                        image.path.name,
                        pdf_path.name,
                        vision_failures,
                        config.vision.max_failures_per_pdf,
                        exc,
                    )
                    if vision_failures >= config.vision.max_failures_per_pdf:
                        LOGGER.warning(
                            "Vision failures reached threshold for %s, skip remaining images.",
                            pdf_path.name,
                        )
                        break
                    continue
                if analysis.keep and analysis.level == "level2":
                    processed.level2_images.append((image, analysis))
                elif analysis.keep and analysis.level == "level3" and config.vision.include_level3:
                    processed.level3_images.append((image, analysis))
        return processed

    try:
        processed = _run_with_retries(
            _extract,
            attempts=attempts,
            action=f"PDF extraction for {pdf_path.name}",
        )
        processed_documents.append(processed)
        return {
            "file": pdf_path.name,
            "status": "ok",
            "level2_images": len(processed.level2_images),
            "level3_images": len(processed.level3_images),
        }
    except Exception as exc:
        LOGGER.exception("Failed to process PDF %s", pdf_path)
        return {
            "file": pdf_path.name,
            "status": "failed",
            "error": str(exc),
        }


def _run_with_retries(fn: Callable[[], Any], *, attempts: int, action: str) -> Any:
    last_error: Exception | None = None
    for index in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            if index == attempts - 1:
                break
            LOGGER.warning(
                "%s failed on attempt %s/%s: %s",
                action,
                index + 1,
                attempts,
                exc,
            )
    if last_error is None:
        raise RuntimeError(f"{action} failed with unknown error")
    raise last_error


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"items": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Manifest at %s is unreadable, start from empty state.", path)
        return {"items": []}
    if not isinstance(payload, dict):
        return {"items": []}
    if not isinstance(payload.get("items"), list):
        payload["items"] = []
    return payload


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _item_record_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for item in manifest.get("items", []):
        if isinstance(item, dict) and isinstance(item.get("item_key"), str):
            records[item["item_key"]] = item
    return records


def _upsert_item_record(manifest: dict[str, Any], record: dict[str, Any]) -> None:
    items = manifest.setdefault("items", [])
    for idx, existing in enumerate(items):
        if isinstance(existing, dict) and existing.get("item_key") == record.get("item_key"):
            items[idx] = record
            return
    items.append(record)


def _load_cached_document(pdf_path: Path, item_output_dir: Path) -> ProcessedDocument | None:
    marker_path = item_output_dir / "marker" / f"{pdf_path.stem}.marker.md"
    if not marker_path.exists():
        return None

    try:
        markdown = marker_path.read_text(encoding="utf-8")
    except OSError:
        return None

    marker_result = MarkerResult(
        pdf_path=pdf_path,
        markdown_path=marker_path,
        markdown=markdown,
        metadata={},
        images=[],
        abstract=None,
        conclusion=None,
    )
    return ProcessedDocument(marker=marker_result)


def _is_pdf_completed_on_disk(pdf_path: Path, item_output_dir: Path) -> bool:
    marker_path = item_output_dir / "marker" / f"{pdf_path.stem}.marker.md"
    return marker_path.exists()


def _build_item_failure_record(
    *,
    item: ZoteroItem,
    output_dir_name: str,
    previous_item_record: dict[str, Any] | None,
    gpu_id: int | None,
    error: Exception,
) -> dict[str, Any]:
    previous_pdfs = []
    if isinstance(previous_item_record, dict):
        raw_pdfs = previous_item_record.get("pdfs", [])
        if isinstance(raw_pdfs, list):
            previous_pdfs = [entry for entry in raw_pdfs if isinstance(entry, dict)]

    return {
        "item_key": item.item_key,
        "citation_key": item.citation_key,
        "title": item.title,
        "year": item.year,
        "output_dir": output_dir_name,
        "gpu_id": gpu_id,
        "pdfs": previous_pdfs,
        "status": "failed",
        "pdf_status": "present" if item.pdf_files else "missing",
        "error": str(error),
    }


def _create_progress(total: int, enabled: bool) -> _ProgressHandle:
    if not enabled or total <= 0:
        return _ProgressHandle(enabled=False)

    try:
        from tqdm import tqdm
    except ImportError:
        LOGGER.warning("tqdm not installed, fallback to logging-only progress.")
        return _ProgressHandle(enabled=False)

    bar = tqdm(total=total, desc="KES papers", unit="paper")
    return _ProgressHandle(enabled=True, _bar=bar)


def _create_item_progress(total: int, enabled: bool, citation_key: str) -> _ProgressHandle:
    if not enabled or total <= 0:
        return _ProgressHandle(enabled=False)

    try:
        from tqdm import tqdm
    except ImportError:
        return _ProgressHandle(enabled=False)

    bar = tqdm(
        total=total,
        desc=f"PDFs {citation_key[:24]}",
        unit="pdf",
        position=1,
        leave=False,
    )
    return _ProgressHandle(enabled=True, _bar=bar)


def _sample_items(items: list[ZoteroItem], sample_size: int | None) -> list[ZoteroItem]:
    if sample_size is None:
        return items
    if sample_size >= len(items):
        return items
    sampled = items[:sample_size]
    LOGGER.info("Sample mode enabled: selected %s/%s papers.", len(sampled), len(items))
    return sampled


def _resolve_worker_count(config: AppConfig, item_count: int) -> int:
    if item_count <= 0:
        return 1

    requested = max(1, int(config.run.parallel_workers))
    effective = min(requested, item_count)

    if config.run.gpu_devices:
        max_by_gpu = len(config.run.gpu_devices)
        if effective > max_by_gpu:
            LOGGER.warning(
                "parallel_workers=%s exceeds gpu_devices count=%s; use %s workers to avoid GPU oversubscription.",
                requested,
                max_by_gpu,
                max_by_gpu,
            )
        effective = min(effective, max_by_gpu)

    return max(1, effective)


def _select_gpu_id(config: AppConfig, item_index: int) -> int | None:
    devices = config.run.gpu_devices
    if not devices:
        return None
    if config.run.gpu_mode == "round-robin":
        return devices[item_index % len(devices)]
    return devices[0]


def _activate_gpu(gpu_id: int | None) -> None:
    if gpu_id is None:
        return
    try:
        import torch
    except Exception:
        LOGGER.warning("GPU %s selected but torch is not importable.", gpu_id)
        return

    if not torch.cuda.is_available():
        LOGGER.warning("GPU %s selected but CUDA is unavailable.", gpu_id)
        return

    try:
        torch.cuda.set_device(gpu_id)
    except Exception as exc:
        LOGGER.warning("Failed to set CUDA device %s: %s", gpu_id, exc)


def _write_run_stats(output_root: Path, scanned_items: list[ZoteroItem], manifest: dict[str, Any]) -> None:
    with_pdf = [item for item in scanned_items if item.pdf_files]
    without_pdf = [item for item in scanned_items if not item.pdf_files]

    failed_pdf = 0
    ok_pdf = 0
    for entry in manifest.get("items", []):
        if not isinstance(entry, dict):
            continue
        for pdf in entry.get("pdfs", []):
            if not isinstance(pdf, dict):
                continue
            if pdf.get("status") == "ok":
                ok_pdf += 1
            elif pdf.get("status") == "failed":
                failed_pdf += 1

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "total_items": len(scanned_items),
        "total_pdfs_detected": sum(len(item.pdf_files) for item in scanned_items),
        "items_with_pdf_count": len(with_pdf),
        "items_without_pdf_count": len(without_pdf),
        "items_with_pdf": [
            {
                "item_key": item.item_key,
                "citation_key": item.citation_key,
                "pdf_count": len(item.pdf_files),
            }
            for item in with_pdf
        ],
        "items_without_pdf": [
            {
                "item_key": item.item_key,
                "citation_key": item.citation_key,
            }
            for item in without_pdf
        ],
        "processed_items": len(manifest.get("items", [])),
        "pdf_ok": ok_pdf,
        "pdf_failed": failed_pdf,
        "unfinished_units_count": len(_collect_unfinished_units(scanned_items, manifest)),
    }

    (output_root / "run_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_unfinished_units(output_root: Path, scanned_items: list[ZoteroItem], manifest: dict[str, Any]) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "unfinished_units": _collect_unfinished_units(scanned_items, manifest),
    }
    (output_root / "unfinished_units.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_checkpoint_outputs(output_root: Path, scanned_items: list[ZoteroItem], manifest: dict[str, Any]) -> None:
    _write_unfinished_units(output_root, scanned_items, manifest)
    _write_run_stats(output_root, scanned_items, manifest)


def _collect_unfinished_units(scanned_items: list[ZoteroItem], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    scanned_by_key = {item.item_key: item for item in scanned_items}
    manifest_items = {
        entry.get("item_key"): entry
        for entry in manifest.get("items", [])
        if isinstance(entry, dict) and isinstance(entry.get("item_key"), str)
    }

    for item_key, item in scanned_by_key.items():
        if item_key not in manifest_items:
            units.append(
                {
                    "scope": "item",
                    "item_key": item.item_key,
                    "citation_key": item.citation_key,
                    "status": "missing-in-manifest",
                    "reason": "item not present in manifest",
                }
            )

    for item_key, entry in manifest_items.items():
        item_status = str(entry.get("status") or "")
        if item_status in {"failed", "partial-failure"}:
            units.append(
                {
                    "scope": "item",
                    "item_key": item_key,
                    "citation_key": entry.get("citation_key", ""),
                    "status": item_status,
                    "reason": entry.get("error", "item has failed or partially failed PDFs"),
                }
            )

        for pdf_entry in entry.get("pdfs", []):
            if not isinstance(pdf_entry, dict):
                continue
            pdf_status = str(pdf_entry.get("status") or "")
            if pdf_status == "ok":
                continue
            units.append(
                {
                    "scope": "pdf",
                    "item_key": item_key,
                    "citation_key": entry.get("citation_key", ""),
                    "file": pdf_entry.get("file", ""),
                    "status": pdf_status or "unknown",
                    "reason": pdf_entry.get("error", "pdf not completed"),
                }
            )

    return units


def _assign_output_dir_names(items: list[ZoteroItem]) -> dict[str, str]:
    assigned: dict[str, str] = {}
    seen: set[str] = set()
    for item in items:
        base = item.citation_key
        candidate = base
        if candidate in seen:
            candidate = f"{base}-{item.item_key.lower()}"
        seen.add(candidate)
        assigned[item.item_key] = candidate
    return assigned


def _write_catalog_index(index_root: Path, catalog_rows: list[dict[str, str]]) -> None:
    rows = sorted(catalog_rows, key=lambda row: row.get("citation_key", ""))
    (index_root / "index.md").write_text(render_catalog_index(rows), encoding="utf-8")


def _write_item_indexes(item_output_dir: Path, markdown: str, output_dir_name: str) -> None:
    (item_output_dir / "index.md").write_text(markdown, encoding="utf-8")
    (item_output_dir / f"{output_dir_name}.index.md").write_text(markdown, encoding="utf-8")


def _display_pdf_status(value: Any) -> str:
    if value == "present":
        return "存在"
    if value == "missing":
        return "缺失"
    return str(value or "unknown")


def _extract_marker_result(
    *,
    pdf_path: Path,
    item_output_dir: Path,
    marker: MarkerExtractor,
    suppress_internal_progress: bool,
) -> MarkerResult:
    if not suppress_internal_progress:
        return marker.extract(pdf_path, item_output_dir)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return marker.extract(pdf_path, item_output_dir)