from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kes_for_zotero.config import AppConfig, MarkerSettings, RunSettings, VisionSettings
from kes_for_zotero.marker_pipeline import MarkerExtractor
from kes_for_zotero.models import ExtractedImage, ZoteroItem
from kes_for_zotero.pipeline import _resolve_worker_count, run_pipeline


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2p6n8AAAAASUVORK5CYII="
)


class FakeRendered:
    def __init__(self) -> None:
        self.markdown = "# Title\n\n## Abstract\nA short abstract.\n\n![fig](fig1.png)"
        self.metadata = {"page_stats": [{"page": 1}]}
        self.images = {"fig1.png": PNG_1X1}


class MarkerPipelineTests(unittest.TestCase):
    def test_marker_extract_writes_markdown_and_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path = root / "doc.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%fake")
            output_dir = root / "out"

            extractor = MarkerExtractor(MarkerSettings())
            extractor._get_converter = lambda: (lambda _pdf: FakeRendered())  # type: ignore[method-assign]

            result = extractor.extract(pdf_path, output_dir)

            self.assertTrue(result.markdown_path.exists())
            self.assertIn("Abstract", result.markdown)
            self.assertEqual(len(result.images), 1)
            self.assertTrue((output_dir / result.images[0].relative_path).exists())

    def test_candidate_images_apply_fine_grained_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_a = root / "a.png"
            image_b = root / "b.png"
            image_small = root / "small.png"
            image_logo = root / "logo.png"

            for path in (image_a, image_b, image_small, image_logo):
                path.write_bytes(PNG_1X1)

            extractor = MarkerExtractor(MarkerSettings())
            vision = VisionSettings(
                min_image_area=100,
                min_short_side=10,
                max_candidate_images=10,
                min_aspect_ratio=1.0,
                max_aspect_ratio=3.0,
                deduplicate_images=True,
                drop_context_keywords=("logo",),
                prefer_context_keywords=("figure", "architecture"),
            )

            images = [
                ExtractedImage(
                    path=image_a,
                    relative_path="assets/a.png",
                    width=50,
                    height=30,
                    context_excerpt="Figure 1 architecture overview",
                ),
                ExtractedImage(
                    path=image_b,
                    relative_path="assets/b.png",
                    width=60,
                    height=30,
                    context_excerpt="Figure 2 similar plot",
                ),
                ExtractedImage(
                    path=image_small,
                    relative_path="assets/small.png",
                    width=8,
                    height=8,
                    context_excerpt="Figure tiny",
                ),
                ExtractedImage(
                    path=image_logo,
                    relative_path="assets/logo.png",
                    width=60,
                    height=30,
                    context_excerpt="institution logo",
                ),
            ]

            class ResultStub:
                def __init__(self, result_images: list[ExtractedImage]) -> None:
                    self.images = result_images

            filtered = extractor.candidate_images(ResultStub(images), vision)  # type: ignore[arg-type]

            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0].path.name, "a.png")


class PipelineRunModeTests(unittest.TestCase):
    def _base_config(self, storage_root: Path, output_root: Path, strict_fail: bool) -> AppConfig:
        return AppConfig(
            storage_root=storage_root,
            output_root=output_root,
            marker=MarkerSettings(),
            vision=VisionSettings(enabled=False),
            run=RunSettings(
                resume=True,
                show_progress=False,
                retry_attempts=0,
                force_reprocess=False,
                strict_fail=strict_fail,
            ),
        )

    def test_resolve_worker_count_caps_by_gpu_devices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._base_config(root / "storage", root / "output", strict_fail=False)
            config.run.parallel_workers = 4
            config.run.gpu_devices = (0, 1)

            self.assertEqual(_resolve_worker_count(config, item_count=10), 2)

    def test_resolve_worker_count_caps_by_item_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._base_config(root / "storage", root / "output", strict_fail=False)
            config.run.parallel_workers = 8
            config.run.gpu_devices = ()

            self.assertEqual(_resolve_worker_count(config, item_count=3), 3)

    def test_run_pipeline_strict_fail_raises_when_pdf_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "storage"
            item_dir = storage / "ITEM001"
            item_dir.mkdir(parents=True)
            pdf = item_dir / "paper.pdf"
            pdf.write_bytes(b"%PDF-1.4\n%fake")

            output = root / "output"
            output.mkdir(parents=True)

            item = ZoteroItem(
                item_key="ITEM001",
                citation_key="doe2024paper",
                title="Paper Title",
                year="2024",
                first_author="doe",
                item_dir=item_dir,
                pdf_files=[pdf],
                related_files=[],
            )
            config = self._base_config(storage, output, strict_fail=True)

            class FailingMarker:
                def __init__(self, _settings: MarkerSettings) -> None:
                    pass

                def extract(self, _pdf_path: Path, _item_output_dir: Path):
                    raise RuntimeError("boom")

                def candidate_images(self, _result, _vision):
                    return []

            with mock.patch("kes_for_zotero.pipeline.scan_storage", return_value=[item]), mock.patch(
                "kes_for_zotero.pipeline.MarkerExtractor", FailingMarker
            ):
                with self.assertRaises(RuntimeError):
                    run_pipeline(config)

    def test_run_pipeline_resume_skips_completed_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "storage"
            item_dir = storage / "ITEM002"
            item_dir.mkdir(parents=True)
            pdf = item_dir / "paper.pdf"
            pdf.write_bytes(b"%PDF-1.4\n%fake")

            output = root / "output"
            marker_dir = output / "papper" / "doe2024cached" / "marker"
            marker_dir.mkdir(parents=True, exist_ok=True)
            (marker_dir / "paper.marker.md").write_text("# Cached marker content", encoding="utf-8")

            manifest = {
                "items": [
                    {
                        "item_key": "ITEM002",
                        "status": "ok",
                        "pdfs": [{"file": "paper.pdf", "status": "ok", "level2_images": 0, "level3_images": 0}],
                    }
                ]
            }
            (output / "manifest.json").write_text(str(manifest).replace("'", '"'), encoding="utf-8")

            item = ZoteroItem(
                item_key="ITEM002",
                citation_key="doe2024cached",
                title="Cached Paper",
                year="2024",
                first_author="doe",
                item_dir=item_dir,
                pdf_files=[pdf],
                related_files=[],
            )
            config = self._base_config(storage, output, strict_fail=False)

            class ShouldNotRunMarker:
                def __init__(self, _settings: MarkerSettings) -> None:
                    pass

                def extract(self, _pdf_path: Path, _item_output_dir: Path):
                    raise AssertionError("resume should skip already successful pdf")

                def candidate_images(self, _result, _vision):
                    return []

            with mock.patch("kes_for_zotero.pipeline.scan_storage", return_value=[item]), mock.patch(
                "kes_for_zotero.pipeline.MarkerExtractor", ShouldNotRunMarker
            ):
                result_manifest = run_pipeline(config)

            self.assertEqual(result_manifest["items"][0]["status"], "ok")
            self.assertTrue((output / "papper" / "doe2024cached" / "index.md").exists())
            self.assertTrue((output / "papper" / "doe2024cached" / "doe2024cached.index.md").exists())

    def test_run_pipeline_builds_index_for_item_without_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "storage"
            item_dir = storage / "ITEM003"
            item_dir.mkdir(parents=True)

            output = root / "output"
            output.mkdir(parents=True)

            item = ZoteroItem(
                item_key="ITEM003",
                citation_key="doe2024nopdf",
                title="No PDF Paper",
                year="2024",
                first_author="doe",
                item_dir=item_dir,
                pdf_files=[],
                related_files=[],
            )
            config = self._base_config(storage, output, strict_fail=False)

            with mock.patch("kes_for_zotero.pipeline.scan_storage", return_value=[item]):
                result_manifest = run_pipeline(config)

            self.assertEqual(result_manifest["items"][0]["status"], "indexed-only")
            self.assertTrue((output / "papper" / "doe2024nopdf" / "index.md").exists())
            self.assertTrue((output / "papper" / "doe2024nopdf" / "doe2024nopdf.index.md").exists())
            self.assertTrue((output / "index" / "doe2024nopdf.md").exists())

            run_stats = json.loads((output / "run_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(run_stats["total_items"], 1)
            self.assertEqual(run_stats["items_without_pdf_count"], 1)
            self.assertEqual(run_stats["items_with_pdf_count"], 0)

    def test_run_pipeline_writes_catalog_entry_and_named_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "storage"
            item_dir = storage / "ITEM004"
            item_dir.mkdir(parents=True)
            pdf = item_dir / "paper.pdf"
            pdf.write_bytes(b"%PDF-1.4\n%fake")

            output = root / "output"
            output.mkdir(parents=True)

            item = ZoteroItem(
                item_key="ITEM004",
                citation_key="doe2024graph",
                title="Graph Paper",
                year="2024",
                first_author="doe",
                item_dir=item_dir,
                pdf_files=[pdf],
                related_files=[],
            )
            config = self._base_config(storage, output, strict_fail=False)

            class SuccessMarker:
                def __init__(self, _settings: MarkerSettings) -> None:
                    pass

                def extract(self, pdf_path: Path, item_output_dir: Path):
                    marker_dir = item_output_dir / "marker"
                    marker_dir.mkdir(parents=True, exist_ok=True)
                    marker_path = marker_dir / f"{pdf_path.stem}.marker.md"
                    marker_path.write_text("# Marker", encoding="utf-8")
                    from kes_for_zotero.models import MarkerResult

                    return MarkerResult(
                        pdf_path=pdf_path,
                        markdown_path=marker_path,
                        markdown="# Marker",
                        metadata={},
                    )

                def candidate_images(self, _result, _vision):
                    return []

            with mock.patch("kes_for_zotero.pipeline.scan_storage", return_value=[item]), mock.patch(
                "kes_for_zotero.pipeline.MarkerExtractor", SuccessMarker
            ):
                run_pipeline(config)

            named_index = output / "papper" / "doe2024graph" / "doe2024graph.index.md"
            catalog_entry = output / "index" / "doe2024graph.md"
            catalog_index = output / "index" / "index.md"

            self.assertTrue(named_index.exists())
            self.assertTrue(catalog_entry.exists())
            self.assertIn("doe2024graph.index.md", catalog_entry.read_text(encoding="utf-8"))
            self.assertIn("./doe2024graph.md", catalog_index.read_text(encoding="utf-8"))

    def test_run_pipeline_sample_size_limits_processed_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            storage = root / "storage"
            output = root / "output"
            output.mkdir(parents=True)

            item_a_dir = storage / "ITEMA"
            item_b_dir = storage / "ITEMB"
            item_a_dir.mkdir(parents=True)
            item_b_dir.mkdir(parents=True)

            item_a = ZoteroItem(
                item_key="ITEMA",
                citation_key="doe2024a",
                title="A",
                year="2024",
                first_author="doe",
                item_dir=item_a_dir,
                pdf_files=[],
                related_files=[],
            )
            item_b = ZoteroItem(
                item_key="ITEMB",
                citation_key="doe2024b",
                title="B",
                year="2024",
                first_author="doe",
                item_dir=item_b_dir,
                pdf_files=[],
                related_files=[],
            )
            config = self._base_config(storage, output, strict_fail=False)
            config.run.sample_size = 1

            with mock.patch("kes_for_zotero.pipeline.scan_storage", return_value=[item_a, item_b]):
                manifest = run_pipeline(config)

            self.assertEqual(len(manifest["items"]), 1)
            self.assertEqual(manifest["items"][0]["item_key"], "ITEMA")


if __name__ == "__main__":
    unittest.main()
