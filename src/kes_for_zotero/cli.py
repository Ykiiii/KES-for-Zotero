from __future__ import annotations

import argparse
import logging
from pathlib import Path

from kes_for_zotero.config import build_config
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
    parser.add_argument("--log-level", default="INFO", help="Logging level, for example INFO or DEBUG.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = build_config(
        config_path=args.config,
        storage_root=args.storage_root,
        output_root=args.output_root,
        include_level3=True if args.include_level3 else None,
        disable_vision=args.disable_vision,
        item_model=args.model,
    )

    manifest = run_pipeline(config, item_key=args.item_key)
    logging.info("Completed. Processed %s items.", len(manifest.get("items", [])))
    return 0