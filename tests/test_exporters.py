import json
import tempfile
import unittest
from pathlib import Path

from parser_core.domain.models import ParsedBlock
from parser_core.infrastructure.exporters import write_jsonl, write_markdown


class ExporterTests(unittest.TestCase):
    def test_jsonl_and_markdown_are_auditable(self):
        block = ParsedBlock(
            block_id="b1",
            document_id="doc",
            page_no=1,
            page_total=1,
            sequence_no=1,
            content_type="paragraph",
            text="Body",
            section_path=["Section"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "blocks.jsonl"
            markdown = Path(tmp) / "normalized.md"
            write_jsonl([block], jsonl)
            write_markdown([block], markdown)

            self.assertEqual(json.loads(jsonl.read_text(encoding="utf-8")), json.loads(jsonl.read_text(encoding="utf-8")))
            self.assertIn("page_no=1", markdown.read_text(encoding="utf-8"))
            self.assertIn("block_id=b1", markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
