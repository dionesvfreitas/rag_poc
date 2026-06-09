import unittest

from parser_core.application.normalizer import normalize_blocks
from parser_core.application.pipeline import ParserPipelineConfig
from parser_core.application.reports import build_report
from parser_core.domain.models import Chunk, ContentType, DocumentAsset, ParsedBlock, ParsedDocument


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
    source_spans=None,
    source_block_ids=None,
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
        source_block_ids=list(source_block_ids or []),
        source_spans=list(source_spans or []),
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
        linked_asset = {
            **asset,
            "link_strategy": "same_page_nearest_text",
            "link_reason": "figure shares same page and section with nearest text block",
            "link_score": 0.75,
            "linked_by": "chunk_builder",
            "link_evidence": {
                "source_sequence_no": 1,
                "target_sequence_no": 4,
                "block_distance": 3,
                "source_page_no": 1,
                "target_page_no": 1,
                "source_section_path": ["1 DO OBJETO"],
                "target_section_path": ["1 DO OBJETO"],
                "decision": "linked",
                "reason": "same_page_nearest_text",
            },
            "metadata": {"source_block_id": "b1", "target_block_id": "b4"},
        }
        decorative_asset = {
            "asset_id": "fig_002",
            "asset_uri": "images/page_001_figure_002.png",
            "asset_type": "image",
            "page_no": 1,
            "link_strategy": "not_linked_decorative",
            "link_reason": "figure appears in front_matter and was marked decorative",
            "link_score": None,
            "linked_by": "chunk_builder",
            "link_evidence": {
                "source_sequence_no": 2,
                "target_sequence_no": None,
                "block_distance": None,
                "source_page_no": 1,
                "target_page_no": None,
                "source_section_path": ["front_matter"],
                "target_section_path": None,
                "decision": "decorative",
                "reason": "decorative_front_matter_logo",
            },
            "metadata": {"source_block_id": "b2"},
        }
        unlinked_asset = {
            "asset_id": "fig_003",
            "asset_uri": "images/page_002_figure_003.png",
            "asset_type": "image",
            "page_no": 2,
            "link_strategy": "not_linked",
            "link_reason": "figure could not be safely linked",
            "link_score": None,
            "linked_by": "chunk_builder",
            "link_evidence": {
                "source_sequence_no": 3,
                "target_sequence_no": None,
                "block_distance": None,
                "source_page_no": 2,
                "target_page_no": None,
                "source_section_path": ["1 DO OBJETO"],
                "target_section_path": None,
                "decision": "not_linked",
                "reason": "distance_too_large",
            },
            "metadata": {"source_block_id": "b3"},
        }
        document = ParsedDocument(
            document_id="doc",
            source_path="doc.pdf",
            source_name="doc.pdf",
            page_total=2,
            metadata={"assets": [asset, decorative_asset, unlinked_asset]},
            blocks=[
                block(
                    "b1",
                    "",
                    content_type=ContentType.FIGURE.value,
                    section_path=["1 DO OBJETO"],
                    metadata={"related_assets": [linked_asset], "base64_removed": 1},
                ),
                block(
                    "b2",
                    "",
                    content_type=ContentType.FIGURE.value,
                    metadata={"related_assets": [decorative_asset]},
                ),
                block(
                    "b3",
                    "",
                    content_type=ContentType.FIGURE.value,
                    metadata={"related_assets": [unlinked_asset]},
                ),
                block("b4", "Texto sem pagina.", page_no=None, metadata={"base64_removed": 1}),
                block(
                    "b5",
                    "Capa.",
                    section_path=["front_matter"],
                    metadata={"is_front_matter": True},
                ),
            ],
        )
        chunks = [
            chunk(1, "Texto vinculado.", section_path=["1 DO OBJETO"], related_assets=[linked_asset]),
            chunk(
                2,
                "continua no meio.",
                metadata={
                    "split_total": 2,
                    "starts_mid_sentence": True,
                    "hard_split": True,
                },
            ),
            chunk(
                3,
                "Capa.",
                section_path=["front_matter"],
                metadata={"is_front_matter": True},
            ),
        ]

        report = build_report(document, chunks, warnings=["adapter warning"])

        self.assertEqual(report.figures_detected, 3)
        self.assertEqual(report.figures_saved, 3)
        self.assertEqual(report.figures_linked_to_chunks, 1)
        self.assertEqual(report.figures_unlinked, 1)
        self.assertEqual(report.figures_marked_decorative, 1)
        self.assertEqual(
            report.figures_linked_to_chunks
            + report.figures_unlinked
            + report.figures_marked_decorative,
            report.figures_detected,
        )
        self.assertEqual(report.base64_removed, 2)
        self.assertEqual(report.front_matter_blocks, 1)
        self.assertEqual(report.front_matter_chunks, 1)
        self.assertEqual(report.chunks_without_section, 1)
        self.assertEqual(report.chunks_split, 1)
        self.assertEqual(report.chunks_mid_sentence, 1)
        self.assertEqual(report.chunks_hard_split, 1)
        self.assertEqual(report.blocks_without_page, 1)
        self.assertEqual(report.blocks_without_section, 3)
        self.assertIn("adapter warning", report.warnings)
        self.assertTrue(any("could not be safely linked" in item for item in report.warnings))
        self.assertTrue(any("marked decorative" in item for item in report.warnings))
        self.assertTrue(any("chunk(s) without section_path" in item for item in report.warnings))
        self.assertTrue(any("hard character split" in item for item in report.warnings))

    def test_report_separates_asset_metrics_by_type(self):
        image_asset = DocumentAsset(
            asset_id="fig_001",
            asset_uri="images/page_001_figure_001.png",
            asset_type="image",
            page_no=1,
            source_block_id="b1",
        )
        table_asset = DocumentAsset(
            asset_id="tbl_001",
            asset_uri="tables/page_001_table_001.json",
            asset_type="table",
            page_no=1,
            source_block_id="b2",
        )
        unknown_asset = DocumentAsset(
            asset_id="asset_001",
            asset_uri="assets/page_001_asset_001.bin",
            asset_type="vendor-specific",
            page_no=1,
            source_block_id="b3",
        )
        linked_image = {
            **image_asset.to_related_asset(),
            "link_strategy": "same_page_nearest_text",
            "link_reason": "figure shares same page and section with nearest text block",
        }
        document = ParsedDocument(
            document_id="doc",
            source_path="doc.pdf",
            source_name="doc.pdf",
            page_total=1,
            assets=[image_asset, table_asset, unknown_asset],
            blocks=[
                block(
                    "b1",
                    "",
                    content_type=ContentType.FIGURE.value,
                    section_path=["1 DO OBJETO"],
                    metadata={"asset_id": "fig_001", "asset_uri": image_asset.asset_uri, "asset_type": "image"},
                ),
                block(
                    "b2",
                    "",
                    content_type=ContentType.TABLE.value,
                    section_path=["1 DO OBJETO"],
                    metadata={"asset_id": "tbl_001", "asset_uri": table_asset.asset_uri, "asset_type": "table"},
                ),
                block(
                    "b3",
                    "",
                    content_type=ContentType.UNKNOWN.value,
                    section_path=["1 DO OBJETO"],
                    metadata={"asset_id": "asset_001", "asset_uri": unknown_asset.asset_uri, "asset_type": "vendor-specific"},
                ),
            ],
        )
        chunks = [
            chunk(
                1,
                "Texto vinculado.",
                section_path=["1 DO OBJETO"],
                related_assets=[linked_image],
            )
        ]

        report = build_report(document, chunks)

        self.assertEqual(report.assets_detected_total, 3)
        self.assertEqual(report.assets_saved_total, 3)
        self.assertEqual(report.assets_by_type, {"image": 1, "table": 1, "unknown": 1})
        self.assertEqual(report.assets_linked_total, 1)
        self.assertEqual(report.assets_unlinked_total, 2)
        self.assertEqual(report.assets_decorative_total, 0)
        self.assertEqual(report.figures_detected, 1)
        self.assertEqual(report.figures_saved, 1)
        self.assertEqual(report.figures_linked_to_chunks, 1)
        self.assertEqual(report.figures_unlinked, 0)
        self.assertEqual(report.tables_detected, 1)
        self.assertEqual(report.charts_detected, 0)
        self.assertEqual(report.diagrams_detected, 0)
        self.assertEqual(report.unknown_assets_detected, 1)
        self.assertTrue(hasattr(report, "figures_saved"))
        self.assertTrue(any("unknown asset_type" in item for item in report.warnings))

    def test_report_warns_about_chunk_without_source_spans(self):
        document = ParsedDocument(
            document_id="doc",
            source_path="doc.pdf",
            source_name="doc.pdf",
            page_total=1,
            blocks=[block("b1", "Texto auditavel.", section_path=["1 DO OBJETO"])],
        )
        chunks = [chunk(1, "Texto auditavel.", section_path=["1 DO OBJETO"])]

        report = build_report(document, chunks)

        self.assertEqual(report.source_spans_total, 0)
        self.assertEqual(report.chunks_with_source_spans, 0)
        self.assertEqual(report.chunks_without_source_spans, 1)
        self.assertEqual(report.chunks_rebuildable_from_spans, 0)
        self.assertEqual(report.chunks_not_rebuildable_from_spans, 0)
        self.assertTrue(any("without source_spans" in item for item in report.warnings))

    def test_report_warns_about_chunk_not_rebuildable_from_source_spans(self):
        document = ParsedDocument(
            document_id="doc",
            source_path="doc.pdf",
            source_name="doc.pdf",
            page_total=1,
            blocks=[block("b1", "Texto original.", section_path=["1 DO OBJETO"])],
        )
        chunks = [
            chunk(
                1,
                "Texto alterado.",
                section_path=["1 DO OBJETO"],
                source_block_ids=["b1"],
                source_spans=[
                    {
                        "block_id": "b1",
                        "block_start_char": 0,
                        "block_end_char": 15,
                        "chunk_start_char": 0,
                        "chunk_end_char": 15,
                        "separator_before": None,
                        "separator_after": None,
                        "split_index": 0,
                    }
                ],
            )
        ]

        report = build_report(document, chunks)

        self.assertEqual(report.source_spans_total, 1)
        self.assertEqual(report.chunks_with_source_spans, 1)
        self.assertEqual(report.chunks_without_source_spans, 0)
        self.assertEqual(report.chunks_rebuildable_from_spans, 0)
        self.assertEqual(report.chunks_not_rebuildable_from_spans, 1)
        self.assertTrue(any("not rebuildable from source_spans" in item for item in report.warnings))

    def test_report_warns_about_asset_link_without_strategy_or_reason(self):
        incomplete_asset = {
            "asset_id": "fig_001",
            "asset_uri": "images/page_001_figure_001.png",
            "asset_type": "image",
        }
        document = ParsedDocument(
            document_id="doc",
            source_path="doc.pdf",
            source_name="doc.pdf",
            page_total=1,
            blocks=[block("b1", "Texto.", section_path=["1 DO OBJETO"])],
        )
        chunks = [
            chunk(
                1,
                "Texto.",
                section_path=["1 DO OBJETO"],
                related_assets=[incomplete_asset],
                source_block_ids=["b1"],
                source_spans=[
                    {
                        "block_id": "b1",
                        "block_start_char": 0,
                        "block_end_char": 6,
                        "chunk_start_char": 0,
                        "chunk_end_char": 6,
                        "separator_before": None,
                        "separator_after": None,
                        "split_index": 0,
                    }
                ],
            )
        ]

        report = build_report(document, chunks)

        self.assertEqual(report.asset_links_without_strategy, 1)
        self.assertEqual(report.asset_links_without_reason, 1)
        self.assertTrue(any("without link_strategy" in item for item in report.warnings))
        self.assertTrue(any("without link_reason" in item for item in report.warnings))

    def test_report_counts_front_matter_without_section_errors(self):
        document = ParsedDocument(
            document_id="doc",
            source_path="doc.pdf",
            source_name="doc.pdf",
            page_total=1,
            blocks=[
                block(
                    "b1",
                    "Capa.",
                    section_path=["front_matter"],
                    metadata={"is_front_matter": True},
                )
            ],
        )
        chunks = [
            chunk(
                1,
                "Capa.",
                section_path=["front_matter"],
                metadata={"is_front_matter": True},
                source_block_ids=["b1"],
                source_spans=[
                    {
                        "block_id": "b1",
                        "block_start_char": 0,
                        "block_end_char": 5,
                        "chunk_start_char": 0,
                        "chunk_end_char": 5,
                        "separator_before": None,
                        "separator_after": None,
                        "split_index": 0,
                    }
                ],
            )
        ]

        report = build_report(document, chunks)

        self.assertEqual(report.front_matter_blocks, 1)
        self.assertEqual(report.front_matter_chunks, 1)
        self.assertEqual(report.blocks_without_section, 0)
        self.assertEqual(report.chunks_without_section, 0)
        self.assertEqual(report.chunks_rebuildable_from_spans, 1)

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
