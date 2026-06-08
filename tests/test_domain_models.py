import tempfile
import unittest
import json
from dataclasses import asdict
from pathlib import Path

from parser_core.domain.ids import block_id, chunk_id, document_id_for_path
from parser_core.domain.models import AssetType, Chunk, DocumentAsset, ContentType, ParsedBlock


class DomainModelTests(unittest.TestCase):
    def test_content_type_contract_contains_expected_values(self):
        self.assertEqual(ContentType.LIST_ITEM.value, "list_item")
        self.assertEqual(ContentType.TABLE.value, "table")

    def test_asset_type_serializes_as_string(self):
        self.assertEqual(AssetType.IMAGE.value, "image")
        self.assertEqual(json.loads(json.dumps({"asset_type": AssetType.IMAGE}))["asset_type"], "image")

    def test_document_asset_serializes_with_initial_contract(self):
        asset = DocumentAsset(
            asset_id="fig_001",
            asset_uri="images/page_001_figure_001.png",
            asset_type="image",
            source_block_id="b1",
            page_no=1,
            caption="Figura 1 - Fluxo.",
            metadata={"storage_backend": "local"},
        )
        data = asdict(asset)

        self.assertEqual(asset.asset_type, AssetType.IMAGE)
        self.assertEqual(data["asset_type"], AssetType.IMAGE)
        self.assertEqual(json.loads(json.dumps(data))["asset_type"], "image")
        self.assertEqual(data["source_block_id"], "b1")
        self.assertEqual(data["caption"], "Figura 1 - Fluxo.")

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

    def test_related_assets_remain_lightweight_references(self):
        related_asset = {
            "asset_id": "fig_001",
            "asset_uri": "images/page_001_figure_001.png",
            "asset_type": "image",
            "link_strategy": "caption_detected",
            "link_reason": "figure has caption immediately after image",
        }
        chunk = Chunk(
            chunk_id="c",
            document_id="d",
            chunk_no=1,
            text="Body",
            page_start=1,
            page_end=1,
            related_assets=[related_asset],
        )

        self.assertEqual(chunk.related_assets[0]["asset_id"], "fig_001")
        self.assertIsInstance(chunk.related_assets[0], dict)


if __name__ == "__main__":
    unittest.main()
