from __future__ import annotations

import argparse
import logging
from pathlib import Path

from kes_for_zotero.config import build_config
from kes_for_zotero.healthcheck import run_self_check
from kes_for_zotero.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract PDFs and Zotero-related files from a Zotero storage directory into Markdown."
    )
    parser.add_argument("--config", type=Path, help="Path to a JSON config file.")
    parser.add_argument("--storage-root", help="Path to Zotero storage root.")
    parser.add_argument("--output-root", help="Directory for generated Markdown outputs.")
    parser.add_argument("--item-key", help="Only process a single Zotero item directory.")
    parser.add_argument("--model", help="Override both Marker and vision model name.")
    parser.add_argument("--include-level3", action="store_true", help="Also keep Level 3 images.")
    parser.add_argument("--disable-vision", action="store_true", help="Disable vision-based image analysis.")
    parser.add_argument("--self-check", action="store_true", help="Run lightweight preflight checks before pipeline.")
    parser.add_argument("--self-check-only", action="store_true", help="Run lightweight preflight checks and exit.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume from existing manifest.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument(
        "--retry-attempts",
        type=int,
        help="Retry count for failed PDF and vision analysis steps. 0 means no retry.",
    )
    parser.add_argument("--force-reprocess", action="store_true", help="Ignore resume cache and reprocess PDFs.")
    parser.add_argument("--strict-fail", action="store_true", help="Return non-zero when any PDF fails.")
    parser.add_argument("--parallel-workers", type=int, help="Number of papers to process in parallel.")
    parser.add_argument("--sample-size", type=int, help="Only process first N papers from storage scan.")
    parser.add_argument(
        "--gpu-mode",
        choices=("single", "round-robin"),
        help="GPU scheduling mode for multi-GPU hosts.",
    )
    parser.add_argument(
        "--gpu-devices",
        help="Comma-separated CUDA device IDs, for example '0,1,2'.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, for example INFO or DEBUG.")
    return parser


def _parse_gpu_devices(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return ()
    try:
        devices = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError("--gpu-devices must be comma-separated integers, for example 0,1") from exc
    if any(device < 0 for device in devices):
        raise ValueError("--gpu-devices only supports non-negative integers")
    return devices


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        gpu_devices = _parse_gpu_devices(args.gpu_devices)
        config = build_config(
            config_path=args.config,
            storage_root=args.storage_root,
            output_root=args.output_root,
            include_level3=True if args.include_level3 else None,
            disable_vision=args.disable_vision,
            item_model=args.model,
            enable_resume=False if args.no_resume else None,
            enable_progress=False if args.no_progress else None,
            retry_attempts=args.retry_attempts,
            force_reprocess=True if args.force_reprocess else None,
            strict_fail=True if args.strict_fail else None,
            parallel_workers=args.parallel_workers,
            sample_size=args.sample_size,
            gpu_mode=args.gpu_mode,
            gpu_devices=gpu_devices,
        )

        if args.self_check or args.self_check_only:
            all_ok, checks = run_self_check(config)
            for item in checks:
                level = logging.INFO if item.ok else logging.ERROR
                status = "OK" if item.ok else "FAIL"
                logging.log(level, "[self-check] %s | %s | %s", status, item.name, item.detail)
            if args.self_check_only:
                return 0 if all_ok else 1
            if not all_ok:
                logging.error("Self-check failed, aborting pipeline. Use --self-check-only for diagnostics.")
                return 1

        manifest = run_pipeline(config, item_key=args.item_key)
        logging.info("Completed. Processed %s items.", len(manifest.get("items", [])))
        return 0
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())