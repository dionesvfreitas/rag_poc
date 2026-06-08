import tempfile
import unittest
from pathlib import Path

from parser_core.application.chunk_builder import ChunkBuilderConfig, build_chunks
from parser_core.application.normalizer import normalize_blocks
from parser_core.application.pipeline import ParserPipelineConfig
from parser_core.domain.models import ContentType, ParsedBlock
from parser_core.infrastructure.asset_store import LocalAssetStore
from parser_core.infrastructure.docling_adapter import docling_document_to_parsed_document


DATA_IMAGE = "![Image](data:image/png;base64,aGVsbG8=)"


def block(text, label="text", block_id="b1", content_type="unknown", metadata=None):
    return ParsedBlock(
        block_id=block_id,
        document_id="doc",
        page_no=1,
        page_total=1,
        sequence_no=int(block_id[1:]),
        content_type=content_type,
        text=text,
        metadata={"label": label, **(metadata or {})},
    )


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

    def test_chunks_reference_related_assets_without_embedding_figure_text(self):
        figure_asset = {
            "asset_id": "fig_001",
            "asset_uri": "images/page_001_figure_001.png",
            "asset_type": "image",
            "page_no": 1,
        }
        blocks = [
            block(
                "",
                label="picture",
                block_id="b1",
                content_type=ContentType.FIGURE.value,
                metadata={
                    "asset_id": "fig_001",
                    "asset_uri": "images/page_001_figure_001.png",
                    "asset_type": "image",
                    "related_assets": [figure_asset],
                },
            ),
            block("Texto do objeto.", block_id="b2", content_type=ContentType.PARAGRAPH.value),
        ]

        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(min_chunk_chars=1, merge_small_chunks=False),
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Texto do objeto.")
        self.assertEqual(chunks[0].related_assets, [figure_asset])
        self.assertNotIn("data:image", chunks[0].text)
        self.assertNotIn("base64", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
