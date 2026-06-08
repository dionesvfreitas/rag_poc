import tempfile
import unittest
from pathlib import Path

from parser_core.application.assets import document_asset_from_figure_block
from parser_core.application.chunk_builder import ChunkBuilderConfig, build_chunks
from parser_core.application.normalizer import normalize_blocks
from parser_core.application.pipeline import ParserPipelineConfig
from parser_core.domain.models import BoundingBox, ContentType, ParsedBlock
from parser_core.infrastructure.asset_store import LocalAssetStore
from parser_core.infrastructure.docling_adapter import docling_document_to_parsed_document


DATA_IMAGE = "![Image](data:image/png;base64,aGVsbG8=)"
ASSET_LINK_FIELDS = {
    "asset_id",
    "asset_type",
    "asset_uri",
    "page_no",
    "link_strategy",
    "link_reason",
    "link_score",
    "linked_by",
    "link_evidence",
    "metadata",
}


def block(
    text,
    label="text",
    block_id="b1",
    content_type="unknown",
    metadata=None,
    page_no=1,
    section_path=None,
    bbox=None,
):
    return ParsedBlock(
        block_id=block_id,
        document_id="doc",
        page_no=page_no,
        page_total=1,
        sequence_no=int(block_id[1:]),
        content_type=content_type,
        text=text,
        section_title=(section_path or [None])[-1],
        section_path=list(section_path or []),
        bbox=bbox,
        metadata={"label": label, **(metadata or {})},
    )


def asset(asset_id):
    return {
        "asset_id": asset_id,
        "asset_uri": f"images/page_001_{asset_id}.png",
        "asset_type": "image",
        "page_no": 1,
    }


def logo_bbox():
    return BoundingBox(x0=474, y0=824, x1=558, y1=806, coord_origin="BOTTOMLEFT")


class Label:
    value = "picture"


class Provenance:
    page_no = 1


class PictureItem:
    label = Label()
    prov = [Provenance()]

    def export_to_markdown(self, doc=None):
        return DATA_IMAGE


class NativePictureDocument:
    pages = [1]

    def iterate_items(self):
        yield PictureItem(), 0


class AssetTests(unittest.TestCase):
    def assert_audited_related_assets(self, chunk):
        for asset in chunk.related_assets:
            self.assertTrue(ASSET_LINK_FIELDS.issubset(asset))
            self.assertTrue(asset["link_strategy"])
            self.assertTrue(asset["link_reason"])
            self.assertIn("link_evidence", asset)
            self.assertEqual(asset["link_evidence"]["decision"], "linked")

    def test_visual_labels_normalize_to_figure_and_strip_base64_text(self):
        blocks, _ = normalize_blocks(
            [
                block(DATA_IMAGE, label="picture", block_id="b1"),
                block(DATA_IMAGE, label="image", block_id="b2"),
                block(DATA_IMAGE, label="graphic", block_id="b3"),
            ],
            ParserPipelineConfig(),
        )

        self.assertEqual([item.content_type for item in blocks], ["figure", "figure", "figure"])
        self.assertTrue(all("data:image" not in item.text for item in blocks))
        self.assertTrue(all("base64" not in item.text for item in blocks))

    def test_docling_picture_is_saved_as_asset_without_textual_base64(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "doc.pdf"
            source.write_bytes(b"pdf")
            asset_store = LocalAssetStore(Path(tmp) / "images")

            document = docling_document_to_parsed_document(
                NativePictureDocument(),
                source,
                asset_store=asset_store,
            )

            self.assertEqual(len(document.blocks), 1)
            figure = document.blocks[0]
            self.assertEqual(figure.metadata["asset_id"], "fig_001")
            self.assertEqual(figure.metadata["asset_type"], "image")
            self.assertEqual(figure.metadata["asset_uri"], "images/page_001_figure_001.png")
            self.assertEqual(figure.text, "")
            self.assertNotIn("data:image", figure.text)
            self.assertTrue((Path(tmp) / "images" / "page_001_figure_001.png").exists())
            self.assertEqual(document.metadata["assets"][0]["asset_id"], "fig_001")

    def test_figure_metadata_can_be_prepared_as_document_asset(self):
        figure = block(
            "",
            label="picture",
            block_id="b1",
            content_type=ContentType.FIGURE.value,
            page_no=3,
            section_path=["1 DO OBJETO"],
            bbox=logo_bbox(),
            metadata={
                "asset_id": "fig_001",
                "asset_uri": "images/page_003_figure_001.png",
                "asset_type": "image",
                "caption": "Figura 1 - Fluxo.",
                "asset": {
                    "asset_id": "fig_001",
                    "asset_uri": "images/page_003_figure_001.png",
                    "asset_type": "image",
                    "page_no": 3,
                    "metadata": {"storage_backend": "local"},
                },
            },
        )

        asset = document_asset_from_figure_block(figure)

        self.assertEqual(asset.asset_id, "fig_001")
        self.assertEqual(asset.asset_type.value, "image")
        self.assertEqual(asset.asset_uri, "images/page_003_figure_001.png")
        self.assertEqual(asset.source_block_id, "b1")
        self.assertEqual(asset.page_no, 3)
        self.assertEqual(asset.bbox, figure.bbox)
        self.assertEqual(asset.caption, "Figura 1 - Fluxo.")
        self.assertIsNone(asset.ocr_text)
        self.assertEqual(asset.metadata["storage_backend"], "local")

    def test_chunks_reference_related_assets_without_embedding_figure_text(self):
        figure_asset = asset("fig_001")
        blocks = [
            block(
                "",
                label="picture",
                block_id="b1",
                content_type=ContentType.FIGURE.value,
                section_path=["1 DO OBJETO"],
                metadata={
                    "asset_id": "fig_001",
                    "asset_uri": "images/page_001_figure_001.png",
                    "asset_type": "image",
                    "related_assets": [figure_asset],
                },
            ),
            block(
                "Texto do objeto.",
                block_id="b2",
                content_type=ContentType.PARAGRAPH.value,
                section_path=["1 DO OBJETO"],
            ),
        ]

        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(min_chunk_chars=1, merge_small_chunks=False),
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Texto do objeto.")
        self.assertEqual(chunks[0].related_assets[0]["asset_id"], "fig_001")
        self.assertEqual(chunks[0].related_assets[0]["link_strategy"], "same_page_nearest_text")
        self.assertTrue(chunks[0].related_assets[0]["link_reason"])
        self.assertEqual(chunks[0].related_assets[0]["linked_by"], "chunk_builder")
        self.assert_audited_related_assets(chunks[0])
        self.assertNotIn("data:image", chunks[0].text)
        self.assertNotIn("base64", chunks[0].text)

    def test_front_matter_figure_is_marked_decorative_and_not_linked_to_body_chunk(self):
        figure_asset = asset("fig_001")
        figure = block(
            "",
            label="picture",
            block_id="b1",
            content_type=ContentType.FIGURE.value,
            section_path=["front_matter"],
            metadata={
                "is_front_matter": True,
                "related_assets": [figure_asset],
            },
        )
        blocks = [
            figure,
            block(
                "1 DO OBJETO",
                block_id="b2",
                content_type=ContentType.TITLE.value,
                section_path=["1 DO OBJETO"],
                metadata={"label": "section_header"},
            ),
            block(
                "Texto do objeto.",
                block_id="b3",
                content_type=ContentType.PARAGRAPH.value,
                section_path=["1 DO OBJETO"],
            ),
        ]

        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(min_chunk_chars=1, merge_small_chunks=False),
        )

        self.assertTrue(all(not chunk.related_assets for chunk in chunks))
        self.assertEqual(figure.metadata["asset_link_status"], "decorative")
        self.assertEqual(figure.metadata["related_assets"][0]["link_strategy"], "not_linked_decorative")
        self.assertEqual(figure.metadata["related_assets"][0]["link_evidence"]["decision"], "decorative")
        self.assertEqual(figure.metadata["related_assets"][0]["link_evidence"]["reason"], "decorative_front_matter_logo")

    def test_figure_with_immediate_caption_is_linked_with_strategy(self):
        figure_asset = asset("fig_001")
        blocks = [
            block(
                "",
                label="picture",
                block_id="b1",
                content_type=ContentType.FIGURE.value,
                section_path=["1 DO OBJETO"],
                metadata={"related_assets": [figure_asset]},
            ),
            block(
                "Figura 1 - Fluxo do processo.",
                label="caption",
                block_id="b2",
                content_type=ContentType.CAPTION.value,
                section_path=["1 DO OBJETO"],
            ),
        ]

        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(min_chunk_chars=1, merge_small_chunks=False),
        )

        self.assertEqual(chunks[0].related_assets[0]["link_strategy"], "caption_detected")
        self.assertEqual(chunks[0].related_assets[0]["link_score"], 1.0)
        self.assertTrue(chunks[0].related_assets[0]["link_reason"])
        self.assert_audited_related_assets(chunks[0])

    def test_figure_without_confidence_is_not_linked(self):
        figure_asset = asset("fig_001")
        figure = block(
            "",
            label="picture",
            block_id="b1",
            content_type=ContentType.FIGURE.value,
            page_no=1,
            section_path=["1 DO OBJETO"],
            metadata={"related_assets": [figure_asset]},
        )
        blocks = [
            figure,
            block(
                "Texto em outra pagina.",
                block_id="b2",
                content_type=ContentType.PARAGRAPH.value,
                page_no=2,
                section_path=["1 DO OBJETO"],
            ),
        ]

        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(min_chunk_chars=1, merge_small_chunks=False),
        )

        self.assertEqual(chunks[0].related_assets, [])
        self.assertEqual(figure.metadata["asset_link_status"], "unlinked")
        self.assertEqual(figure.metadata["related_assets"][0]["link_strategy"], "not_linked")
        self.assertEqual(figure.metadata["related_assets"][0]["link_evidence"]["reason"], "different_page")

    def test_repeated_small_logo_is_marked_decorative_and_not_related_to_chunks(self):
        first_logo = block(
            "",
            label="picture",
            block_id="b1",
            content_type=ContentType.FIGURE.value,
            page_no=1,
            section_path=["1 DO OBJETO"],
            bbox=logo_bbox(),
            metadata={"related_assets": [asset("fig_001")]},
        )
        second_logo = block(
            "",
            label="picture",
            block_id="b3",
            content_type=ContentType.FIGURE.value,
            page_no=2,
            section_path=["1 DO OBJETO"],
            bbox=logo_bbox(),
            metadata={"related_assets": [asset("fig_002")]},
        )
        blocks = [
            first_logo,
            block("Texto da pagina 1.", block_id="b2", content_type=ContentType.PARAGRAPH.value, section_path=["1 DO OBJETO"]),
            second_logo,
            block("Texto da pagina 2.", block_id="b4", content_type=ContentType.PARAGRAPH.value, page_no=2, section_path=["1 DO OBJETO"]),
        ]

        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(min_chunk_chars=1, merge_small_chunks=False),
        )

        self.assertTrue(all(not chunk.related_assets for chunk in chunks))
        self.assertEqual(first_logo.metadata["asset_link_status"], "decorative")
        self.assertEqual(second_logo.metadata["asset_link_status"], "decorative")
        self.assertEqual(first_logo.metadata["related_assets"][0]["link_evidence"]["reason"], "decorative_repeated_header")

    def test_header_footer_figure_is_not_linked_to_real_chunk(self):
        header_logo = block(
            "",
            label="picture",
            block_id="b1",
            content_type=ContentType.FIGURE.value,
            section_path=["1 DO OBJETO"],
            bbox=logo_bbox(),
            metadata={"related_assets": [asset("fig_001")]},
        )
        blocks = [
            header_logo,
            block("Texto do objeto.", block_id="b2", content_type=ContentType.PARAGRAPH.value, section_path=["1 DO OBJETO"]),
        ]

        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(min_chunk_chars=1, merge_small_chunks=False),
        )

        self.assertEqual(chunks[0].related_assets, [])
        self.assertEqual(header_logo.metadata["asset_link_status"], "decorative")
        self.assertEqual(header_logo.metadata["related_assets"][0]["link_evidence"]["reason"], "decorative_small_logo")

    def test_figure_farther_than_sequence_limit_is_not_linked(self):
        figure = block(
            "",
            label="picture",
            block_id="b1",
            content_type=ContentType.FIGURE.value,
            section_path=["1 DO OBJETO"],
            metadata={"related_assets": [asset("fig_001")]},
        )
        blocks = [
            figure,
            block("Intermediario 1.", block_id="b2", content_type=ContentType.PARAGRAPH.value, section_path=["1 DO OBJETO"]),
            block("Intermediario 2.", block_id="b3", content_type=ContentType.PARAGRAPH.value, section_path=["1 DO OBJETO"]),
            block("Intermediario 3.", block_id="b4", content_type=ContentType.PARAGRAPH.value, section_path=["1 DO OBJETO"]),
            block("Texto distante.", block_id="b5", content_type=ContentType.PARAGRAPH.value, section_path=["1 DO OBJETO"]),
        ]

        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(min_chunk_chars=1, merge_small_chunks=False, max_asset_sequence_distance=0),
        )

        self.assertTrue(all(not chunk.related_assets for chunk in chunks))
        self.assertEqual(figure.metadata["asset_link_status"], "unlinked")
        self.assertEqual(figure.metadata["related_assets"][0]["link_evidence"]["reason"], "distance_too_large")

    def test_figure_with_only_previous_text_is_not_linked_by_nearest_text(self):
        figure = block(
            "",
            label="picture",
            block_id="b2",
            content_type=ContentType.FIGURE.value,
            section_path=["1 DO OBJETO"],
            metadata={"related_assets": [asset("fig_001")]},
        )
        blocks = [
            block("Texto anterior.", block_id="b1", content_type=ContentType.PARAGRAPH.value, section_path=["1 DO OBJETO"]),
            figure,
        ]

        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(min_chunk_chars=1, merge_small_chunks=False),
        )

        self.assertEqual(chunks[0].related_assets, [])
        self.assertEqual(figure.metadata["asset_link_status"], "unlinked")
        self.assertEqual(figure.metadata["related_assets"][0]["link_evidence"]["reason"], "no_posterior_text_candidate")


if __name__ == "__main__":
    unittest.main()
