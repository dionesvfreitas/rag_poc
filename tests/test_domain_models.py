import tempfile
import unittest
from pathlib import Path

from parser_core.domain.ids import block_id, chunk_id, document_id_for_path
from parser_core.domain.models import Chunk, ContentType, ParsedBlock


class DomainModelTests(unittest.TestCase):
    def test_content_type_contract_contains_expected_values(self):
        self.assertEqual(ContentType.LIST_ITEM.value, "list_item")
        self.assertEqual(ContentType.TABLE.value, "table")

    def test_ids_are_sha256_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.pdf"
            path.write_bytes(b"pdf")
            first = document_id_for_path(path)
            second = document_id_for_path(path)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertEqual(len(block_id("doc", 1, 1)), 64)
        self.assertEqual(len(chunk_id("doc", 1)), 64)

    def test_models_are_plain_domain_objects(self):
        block = ParsedBlock(
            block_id="b",
            document_id="d",
            page_no=1,
            page_total=1,
            sequence_no=1,
            content_type="paragraph",
            text="Body",
        )
        chunk = Chunk(
            chunk_id="c",
            document_id="d",
            chunk_no=1,
            text=block.text,
            page_start=1,
            page_end=1,
            source_block_ids=[block.block_id],
            content_types=[block.content_type],
        )

        self.assertEqual(chunk.source_block_ids, ["b"])


if __name__ == "__main__":
    unittest.main()
