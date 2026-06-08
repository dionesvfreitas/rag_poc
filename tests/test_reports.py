import unittest

from parser_core.application.normalizer import normalize_blocks
from parser_core.application.pipeline import ParserPipelineConfig
from parser_core.application.reports import build_report
from parser_core.domain.models import Chunk, ContentType, ParsedBlock, ParsedDocument


def block(
    block_id,
    text,
    content_type="paragraph",
    page_no=1,
    section_path=None,
    metadata=None,
):
    return ParsedBlock(
        block_id=block_id,
        document_id="doc",
        page_no=page_no,
        page_total=2,
        sequence_no=int(block_id[1:]),
        content_type=content_type,
        text=text,
        section_title=(section_path or [None])[-1],
        section_path=list(section_path or []),
        metadata=metadata or {},
    )


def chunk(
    chunk_no,
    text,
    section_path=None,
    related_assets=None,
    metadata=None,
):
    return Chunk(
        chunk_id=f"c{chunk_no}",
        document_id="doc",
        chunk_no=chunk_no,
        text=text,
        page_start=1,
        page_end=1,
        section_title=(section_path or [None])[-1],
        section_path=list(section_path or []),
        related_assets=list(related_assets or []),
        metadata=metadata or {},
    )


class ReportTests(unittest.TestCase):
    def test_report_counts_audit_metrics_and_adds_relevant_warnings(self):
        asset = {
            "asset_id": "fig_001",
            "asset_uri": "images/page_001_figure_001.png",
            "asset_type": "image",
            "page_no": 1,
        }
        document = ParsedDocument(
            document_id="doc",
            source_path="doc.pdf",
            source_name="doc.pdf",
            page_total=2,
            metadata={"assets": [asset]},
            blocks=[
                block(
                    "b1",
                    "",
                    content_type=ContentType.FIGURE.value,
                    section_path=["1 DO OBJETO"],
                    metadata={"related_assets": [asset], "base64_removed": 1},
                ),
                block("b2", "", content_type=ContentType.FIGURE.value),
                block("b3", "Texto sem pagina.", page_no=None, metadata={"base64_removed": 1}),
            ],
        )
        chunks = [
            chunk(1, "Texto vinculado.", section_path=["1 DO OBJETO"], related_assets=[asset]),
            chunk(
                2,
                "continua no meio.",
                metadata={
                    "split_total": 2,
                    "starts_mid_sentence": True,
                    "hard_split": True,
                },
            ),
        ]

        report = build_report(document, chunks, warnings=["adapter warning"])

        self.assertEqual(report.figures_detected, 2)
        self.assertEqual(report.figures_saved, 1)
        self.assertEqual(report.figures_linked_to_chunks, 1)
        self.assertEqual(report.base64_removed, 2)
        self.assertEqual(report.chunks_without_section, 1)
        self.assertEqual(report.chunks_split, 1)
        self.assertEqual(report.chunks_mid_sentence, 1)
        self.assertEqual(report.chunks_hard_split, 1)
        self.assertEqual(report.blocks_without_page, 1)
        self.assertEqual(report.blocks_without_section, 2)
        self.assertIn("adapter warning", report.warnings)
        self.assertTrue(any("figure block(s) have no saved asset" in item for item in report.warnings))
        self.assertTrue(any("chunk(s) without section_path" in item for item in report.warnings))
        self.assertTrue(any("hard character split" in item for item in report.warnings))

    def test_normalizer_records_base64_removed_for_report_audit(self):
        blocks, _ = normalize_blocks(
            [
                block(
                    "b1",
                    "![Image](data:image/png;base64,aGVsbG8=)",
                    metadata={"label": "picture"},
                ),
            ],
            ParserPipelineConfig(),
        )

        self.assertEqual(blocks[0].metadata["base64_removed"], 1)
        self.assertEqual(blocks[0].text, "")


if __name__ == "__main__":
    unittest.main()
