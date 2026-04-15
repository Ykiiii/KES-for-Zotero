from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kes_for_zotero.config import build_config
from kes_for_zotero.marker_pipeline import MarkerExtractor
from kes_for_zotero.zotero_storage import scan_storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Marker runtime and parse a single Zotero PDF.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.example.json")
    parser.add_argument("--item-key", help="Specific Zotero item key to diagnose.")
    parser.add_argument("--pdf-name", help="Specific PDF filename inside the item directory.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PROJECT_ROOT / "output" / "marker_diagnostics.json",
        help="Where to write the JSON diagnostic report.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=PROJECT_ROOT / "output" / "marker_diagnostics_run.log",
        help="Where to write the plain-text diagnostic log.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    logger = StepLogger(args.log_path)
    started_at = time.perf_counter()

    report: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config),
        "item_key": args.item_key,
        "pdf_name": args.pdf_name,
        "log_path": str(args.log_path),
    }
    try:
        logger.log("stage=packages start")
        report["packages"] = package_versions()
        logger.log(f"stage=packages done packages={report['packages']}")

        logger.log("stage=imports start")
        report["imports"] = import_status()
        logger.log(f"stage=imports done imports={report['imports']}")

        logger.log("stage=config start")
        config = build_config(
            config_path=args.config,
            storage_root=None,
            output_root=None,
            include_level3=None,
            disable_vision=True,
            item_model=None,
        )
        report["cache"] = cache_status(config.marker.model_cache_dir)
        logger.log(f"stage=config done output_root={config.output_root}")

        logger.log("stage=scan_storage start")
        items = scan_storage(config.storage_root, args.item_key)
        logger.log(f"stage=scan_storage done item_count={len(items)}")
        if not items:
            raise RuntimeError(f"No Zotero items found for item_key={args.item_key!r}")

        item = items[0]
        pdf_path = pick_pdf(item.pdf_files, args.pdf_name)
        if pdf_path is None:
            raise RuntimeError(f"No PDF matched pdf_name={args.pdf_name!r}")

        debug_output_dir = config.output_root / f"{item.item_key}_debug"
        extractor = MarkerExtractor(config.marker)

        report["selected_item"] = item.item_key
        report["selected_pdf"] = str(pdf_path)
        report["debug_output_dir"] = str(debug_output_dir)
        logger.log(f"stage=selection done item={item.item_key} pdf={pdf_path.name}")

        logger.log("stage=extract start")
        extract_started_at = time.perf_counter()
        ticker = ProgressTicker(logger, label="stage=extract running", interval_seconds=20)
        ticker.start()
        try:
            result = extractor.extract(pdf_path, debug_output_dir)
        finally:
            ticker.stop()
        extract_elapsed = round(time.perf_counter() - extract_started_at, 3)
        logger.log(f"stage=extract done seconds={extract_elapsed} markdown={result.markdown_path}")

        report["status"] = "ok"
        report["marker_result"] = {
            "markdown_path": str(result.markdown_path),
            "markdown_exists": result.markdown_path.exists(),
            "markdown_chars": len(result.markdown),
            "image_count": len(result.images),
            "image_paths": [image.relative_path for image in result.images[:20]],
            "abstract_found": bool(result.abstract),
            "conclusion_found": bool(result.conclusion),
            "metadata_keys": sorted(result.metadata.keys()),
            "extract_seconds": extract_elapsed,
        }
        logger.log("stage=summary done status=ok")
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        logger.log(f"stage=failed error={type(exc).__name__}: {exc}")
        logger.log(report["traceback"])
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
        write_report(args.report_path, report)
        logger.log(f"stage=report written path={args.report_path} status={report.get('status')}")
        logger.close()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 1


def package_versions() -> dict[str, str]:
    names = ["torch", "torchvision", "torchaudio", "transformers", "marker-pdf", "surya-ocr"]
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def import_status() -> dict[str, str]:
    modules = [
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "marker",
    ]
    result: dict[str, str] = {}
    for name in modules:
        try:
            importlib.import_module(name)
            result[name] = "ok"
        except Exception as exc:
            result[name] = f"fail: {type(exc).__name__}: {exc}"
    return result


def cache_status(model_cache_dir: Path) -> dict[str, Any]:
    cache_root = model_cache_dir.parent
    directories = {
        "model_cache_dir": model_cache_dir,
        "huggingface_root": cache_root / "huggingface",
        "huggingface_hub": cache_root / "huggingface" / "hub",
        "torch_root": cache_root / "torch",
    }

    result: dict[str, Any] = {}
    for name, path in directories.items():
        result[name] = {
            "path": str(path),
            "exists": path.exists(),
            "children": sorted(child.name for child in path.iterdir())[:20] if path.exists() else [],
        }
    return result


def pick_pdf(pdf_files: list[Path], pdf_name: str | None) -> Path | None:
    if not pdf_files:
        return None
    if pdf_name is None:
        return pdf_files[0]
    for pdf_path in pdf_files:
        if pdf_path.name == pdf_name:
            return pdf_path
    return None


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


class StepLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class ProgressTicker:
    def __init__(self, logger: StepLogger, *, label: str, interval_seconds: int = 20) -> None:
        self.logger = logger
        self.label = label
        self.interval_seconds = max(5, interval_seconds)
        self._event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def start(self) -> None:
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._event.wait(self.interval_seconds):
            elapsed = round(time.perf_counter() - self._started_at, 1)
            self.logger.log(f"{self.label} elapsed={elapsed}s")


if __name__ == "__main__":
    raise SystemExit(main())