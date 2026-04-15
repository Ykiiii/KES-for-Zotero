from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kes_for_zotero.markdown_writer import render_item_markdown
from kes_for_zotero.models import MarkerResult, ProcessedDocument, ZoteroItem


class MarkdownWriterStorageTests(unittest.TestCase):
    def test_index_does_not_embed_full_marker_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item_dir = root / "ITEM001"
            item_dir.mkdir(parents=True)

            pdf_path = item_dir / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.4")

            marker_path = item_dir / "marker" / "paper.marker.md"
            marker_path.parent.mkdir(parents=True)
            marker_text = "VERY_LONG_MARKER_CONTENT_SHOULD_NOT_BE_IN_INDEX"
            marker_path.write_text(marker_text, encoding="utf-8")

            item = ZoteroItem(
                item_key="ITEM001",
                citation_key="doe2024paper",
                title="Paper Title",
                year="2024",
                first_author="doe",
                item_dir=item_dir,
                pdf_files=[pdf_path],
                related_files=[],
            )
            marker_result = MarkerResult(
                pdf_path=pdf_path,
                markdown_path=marker_path,
                markdown=marker_text,
                metadata={},
            )
            document = ProcessedDocument(marker=marker_result)

            rendered = render_item_markdown(item, [document], item_dir)

            self.assertIn("Marker 原始提取文件", rendered)
            self.assertIn("marker/paper.marker.md", rendered)
            self.assertNotIn(marker_text, rendered)


if __name__ == "__main__":
    unittest.main()
