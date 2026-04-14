from __future__ import annotations

import json
import logging

from kes_for_zotero.config import AppConfig
from kes_for_zotero.marker_pipeline import MarkerExtractor
from kes_for_zotero.markdown_writer import render_item_markdown
from kes_for_zotero.models import ProcessedDocument
from kes_for_zotero.vision_llm import OllamaVisionClient
from kes_for_zotero.zotero_storage import scan_storage

LOGGER = logging.getLogger(__name__)


def run_pipeline(config: AppConfig, item_key: str | None = None) -> dict:
    config.output_root.mkdir(parents=True, exist_ok=True)

    items = scan_storage(config.storage_root, item_key)
    marker = MarkerExtractor(config.marker)
    vision = OllamaVisionClient(config.vision) if config.vision.enabled else None

    manifest: dict[str, list[dict]] = {"items": []}

    for item in items:
        LOGGER.info("Processing item %s", item.item_key)
        item_output_dir = config.output_root / item.item_key
        item_output_dir.mkdir(parents=True, exist_ok=True)

        processed_documents: list[ProcessedDocument] = []
        item_record = {
            "item_key": item.item_key,
            "pdfs": [],
            "status": "ok",
        }

        for pdf_path in item.pdf_files:
            try:
                marker_result = marker.extract(pdf_path, item_output_dir)
                processed = ProcessedDocument(marker=marker_result)
                if vision is not None:
                    for image in marker.candidate_images(marker_result, config.vision):
                        try:
                            analysis = vision.analyze_image(image, pdf_path.name)
                        except Exception as exc:
                            LOGGER.warning(
                                "Skipping image %s from %s after vision analysis failure: %s",
                                image.path.name,
                                pdf_path.name,
                                exc,
                            )
                            item_record["status"] = "partial-failure"
                            continue

                        if analysis.keep and analysis.level == "level2":
                            processed.level2_images.append((image, analysis))
                        elif analysis.keep and analysis.level == "level3" and config.vision.include_level3:
                            processed.level3_images.append((image, analysis))
                processed_documents.append(processed)
                item_record["pdfs"].append(
                    {
                        "file": pdf_path.name,
                        "status": "ok",
                        "level2_images": len(processed.level2_images),
                        "level3_images": len(processed.level3_images),
                    }
                )
            except Exception as exc:
                LOGGER.exception("Failed to process PDF %s", pdf_path)
                item_record["status"] = "partial-failure"
                item_record["pdfs"].append(
                    {
                        "file": pdf_path.name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        markdown = render_item_markdown(item, processed_documents)
        (item_output_dir / "index.md").write_text(markdown, encoding="utf-8")
        manifest["items"].append(item_record)

    config.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest