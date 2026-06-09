import tempfile
import unittest
import json
from dataclasses import asdict
from pathlib import Path

from parser_core.domain.ids import block_id, chunk_id, document_id_for_path
from parser_core.domain.models import AssetType, BoundingBox, Chunk, DocumentAsset, ContentType, ParsedBlock, ParsedDocument, to_dict


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

    def test_document_asset_supports_canonical_asset_types(self):
        for asset_type in ["image", "figure", "table", "chart", "diagram", "unknown"]:
            asset = DocumentAsset(
                asset_id=f"{asset_type}_001",
                asset_uri=f"assets/{asset_type}_001",
                asset_type=asset_type,
            )

            self.assertEqual(asset.asset_type.value, asset_type)

    def test_document_asset_normalizes_unknown_types_bbox_and_metadata(self):
        class ExternalObject:
            def __str__(self):
                return "external-object"

        asset = DocumentAsset(
            asset_id="asset_001",
            asset_uri="assets/asset_001.bin",
            asset_type="docling-special",
            source_block_id="b1",
            page_no=2,
            bbox={"x0": 1, "y0": 2, "x1": 3, "y1": 4, "coord_origin": "BOTTOMLEFT"},
            caption="Asset caption",
            ocr_text="OCR text",
            metadata={"external": ExternalObject(), "path": Path("asset.bin")},
        )
        data = to_dict(asset)

        self.assertEqual(asset.asset_type, AssetType.UNKNOWN)
        self.assertIsInstance(asset.bbox, BoundingBox)
        self.assertEqual(data["bbox"]["x0"], 1)
        self.assertEqual(data["metadata"]["external"], "external-object")
        self.assertEqual(data["metadata"]["path"], "asset.bin")

    def test_parsed_document_preserves_canonical_assets(self):
        asset = DocumentAsset(
            asset_id="fig_001",
            asset_uri="images/page_001_figure_001.png",
            asset_type=AssetType.IMAGE,
            source_block_id="b1",
            page_no=1,
        )
        document = ParsedDocument(
            document_id="doc",
            source_path="doc.pdf",
            source_name="doc.pdf",
            page_total=1,
            assets=[asset],
        )

        self.assertEqual(document.assets[0].asset_id, "fig_001")
        self.assertEqual(to_dict(document)["assets"][0]["asset_type"], "image")

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
        asset = DocumentAsset(
            asset_id="fig_001",
            asset_uri="images/page_001_figure_001.png",
            asset_type="image",
            source_block_id="b1",
            page_no=1,
        )
        related_asset = {
            **asset.to_related_asset(),
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
        self.assertNotIsInstance(chunk.related_assets[0], DocumentAsset)


if __name__ == "__main__":
    unittest.main()
