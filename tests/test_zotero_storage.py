from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kes_for_zotero.zotero_storage import build_citation_key, build_related_file, read_text_preview, scan_storage


class ZoteroStoragePreviewTests(unittest.TestCase):
    def test_zotero_hidden_files_have_text_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info_path = root / ".zotero-ft-info"
            cache_path = root / ".zotero-ft-cache"

            info_text = "Title: Sample Paper\nPages: 12\n"
            cache_text = "Abstract\nThis is cached fulltext."

            info_path.write_text(info_text, encoding="utf-8")
            cache_path.write_text(cache_text, encoding="utf-8")

            info_related = build_related_file(info_path)
            cache_related = build_related_file(cache_path)

            self.assertEqual(info_related.kind, "zotero-fulltext-info")
            self.assertEqual(cache_related.kind, "zotero-fulltext-cache")
            self.assertIsNotNone(info_related.preview)
            self.assertIsNotNone(cache_related.preview)
            self.assertIn("Title: Sample Paper", info_related.preview or "")
            self.assertIn("cached fulltext", cache_related.preview or "")
            self.assertEqual(info_related.structured_fields.get("Title"), "Sample Paper")
            self.assertEqual(info_related.structured_fields.get("Pages"), "12")
            self.assertIn("detected_title", cache_related.structured_fields)
            self.assertIn("line_count", cache_related.structured_fields)

    def test_regular_binary_like_file_still_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(b"\x00\x01\x02")

            preview = read_text_preview(path)
            self.assertIsNone(preview)

    def test_scan_storage_builds_bbt_style_citation_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            item_dir = storage / "ITEM001"
            item_dir.mkdir(parents=True)

            info_path = item_dir / ".zotero-ft-info"
            info_path.write_text(
                "Title: The Graph Neural Network Revolution\n"
                "Author: Jane Doe\n"
                "CreationDate: 2024-03-10\n",
                encoding="utf-8",
            )

            items = scan_storage(storage)

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].citation_key, "doe2024graph")

    def test_build_citation_key_uses_auth_year_shorttitle(self) -> None:
        key = build_citation_key(
            first_author="Smith",
            year="2020",
            title="Deep Learning for Robotics",
            fallback="ABCD1234",
        )
        self.assertEqual(key, "smith2020deep")

    def test_scan_storage_parses_pdf_filename_for_bbt_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            item_dir = storage / "ITEM002"
            item_dir.mkdir(parents=True)

            pdf_path = item_dir / "Xiao 等 - 2018 - Design and evaluation of a 7-DOF cable-driven upper limb exoskeleton.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%fake")

            items = scan_storage(storage)

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].first_author, "xiao")
            self.assertEqual(items[0].year, "2018")
            self.assertEqual(items[0].title, "Design and evaluation of a 7-DOF cable-driven upper limb exoskeleton")
            self.assertEqual(items[0].citation_key, "xiao2018design")

    def test_scan_storage_records_html_and_zotero_files_without_processing_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "storage"
            item_dir = storage / "ITEM003"
            item_dir.mkdir(parents=True)

            (item_dir / "12345678.html").write_text("<html><body>snapshot</body></html>", encoding="utf-8")
            (item_dir / ".zotero-ft-info").write_text(
                "Title: Sample Paper\nAuthor: Jane Doe\nCreationDate: 2024-03-10\n",
                encoding="utf-8",
            )

            items = scan_storage(storage)

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].pdf_files, [])
            self.assertEqual(len(items[0].related_files), 2)
            related_kinds = {related.kind for related in items[0].related_files}
            self.assertIn("snapshot", related_kinds)
            self.assertIn("zotero-fulltext-info", related_kinds)


if __name__ == "__main__":
    unittest.main()
