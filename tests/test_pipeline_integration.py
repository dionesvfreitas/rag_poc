import unittest
import json
from dataclasses import asdict
from pathlib import Path

from parser_core.application.pipeline import ParserPipelineConfig, run_pipeline
from parser_core.domain.models import ParsedBlock, ParsedDocument


def assert_chunk_matches_golden_schema(test_case, shape, schema):
    test_case.assertTrue(all(field in shape for field in schema["required_fields"]))
    test_case.assertTrue(shape["source_spans"])
    for span in shape["source_spans"]:
        for field in schema["source_span_required_fields"]:
            test_case.assertIn(field, span)
    for asset in shape.get("related_assets", []):
        for field in schema["related_asset_required_fields"]:
            test_case.assertIn(field, asset)
            test_case.assertIsNotNone(asset[field])
    if shape.get("metadata", {}).get("is_front_matter"):
        test_case.assertEqual(shape["section_path"], schema["front_matter_section_path"])


class PipelineIntegrationTests(unittest.TestCase):
    def test_pipeline_generates_blocks_chunks_and_report(self):
        document = ParsedDocument(
            document_id="doc",
            source_path="doc.pdf",
            source_name="doc.pdf",
            page_total=1,
            blocks=[
                ParsedBlock(
                    block_id="b1",
                    document_id="doc",
                    page_no=1,
                    page_total=1,
                    sequence_no=1,
                    content_type="unknown",
                    text="1 DO OBJETO",
                    metadata={"label": "section_header"},
                ),
                ParsedBlock(
                    block_id="b2",
                    document_id="doc",
                    page_no=1,
                    page_total=1,
                    sequence_no=2,
                    content_type="unknown",
                    text="1.1 Seleção de pessoas.",
                    metadata={"label": "text"},
                ),
            ],
        )

        parsed, chunks, report = run_pipeline(document, ParserPipelineConfig())

        self.assertEqual(len(parsed.blocks), 2)
        self.assertTrue(chunks)
        self.assertEqual(report.total_blocks, 2)
        self.assertEqual(report.total_chunks, len(chunks))

    def test_pipeline_marks_front_matter_and_keeps_real_section_chunks(self):
        document = ParsedDocument(
            document_id="doc",
            source_path="doc.pdf",
            source_name="doc.pdf",
            page_total=1,
            blocks=[
                ParsedBlock(
                    block_id="b1",
                    document_id="doc",
                    page_no=1,
                    page_total=1,
                    sequence_no=1,
                    content_type="unknown",
                    text="EDITAL 123/2026",
                    metadata={"label": "section_header"},
                ),
                ParsedBlock(
                    block_id="b2",
                    document_id="doc",
                    page_no=1,
                    page_total=1,
                    sequence_no=2,
                    content_type="unknown",
                    text="SECRETARIA DE ADMINISTRACAO",
                    metadata={"label": "section_header"},
                ),
                ParsedBlock(
                    block_id="b3",
                    document_id="doc",
                    page_no=1,
                    page_total=1,
                    sequence_no=3,
                    content_type="unknown",
                    text="1 DO OBJETO",
                    metadata={"label": "section_header"},
                ),
                ParsedBlock(
                    block_id="b4",
                    document_id="doc",
                    page_no=1,
                    page_total=1,
                    sequence_no=4,
                    content_type="unknown",
                    text="Texto do objeto.",
                    metadata={"label": "text"},
                ),
            ],
        )

        parsed, chunks, report = run_pipeline(document, ParserPipelineConfig())

        self.assertEqual(parsed.blocks[0].section_path, ["front_matter"])
        self.assertEqual(parsed.blocks[1].section_path, ["front_matter"])
        self.assertTrue(parsed.blocks[0].metadata["is_front_matter"])
        self.assertEqual(parsed.blocks[2].section_path, ["1 DO OBJETO"])
        self.assertEqual(parsed.blocks[3].section_path, ["1 DO OBJETO"])
        self.assertEqual(chunks[0].section_path, ["front_matter"])
        self.assertTrue(chunks[0].metadata["is_front_matter"])
        self.assertEqual(chunks[-1].section_path, ["1 DO OBJETO"])
        self.assertEqual(report.front_matter_blocks, 2)
        self.assertEqual(report.front_matter_chunks, 1)
        self.assertEqual(report.chunks_without_section, 0)

    def test_chunk_shape_matches_golden_schema(self):
        schema = json.loads(Path("tests/golden/chunk_schema.json").read_text(encoding="utf-8"))
        document = ParsedDocument(
            document_id="doc",
            source_path="doc.pdf",
            source_name="doc.pdf",
            page_total=1,
            blocks=[
                ParsedBlock(
                    block_id="b1",
                    document_id="doc",
                    page_no=1,
                    page_total=1,
                    sequence_no=1,
                    content_type="unknown",
                    text="Body",
                    metadata={"label": "text"},
                ),
            ],
        )

        _, chunks, _ = run_pipeline(document, ParserPipelineConfig())
        shape = asdict(chunks[0])

        assert_chunk_matches_golden_schema(self, shape, schema)

    def test_golden_schema_rejects_related_asset_without_minimum_contract(self):
        schema = json.loads(Path("tests/golden/chunk_schema.json").read_text(encoding="utf-8"))
        shape = {
            "chunk_id": "c1",
            "document_id": "doc",
            "chunk_no": 1,
            "text": "Texto.",
            "page_start": 1,
            "page_end": 1,
            "section_title": "1 DO OBJETO",
            "section_path": ["1 DO OBJETO"],
            "source_block_ids": ["b1"],
            "source_spans": [
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
            "content_types": ["paragraph"],
            "related_assets": [
                {
                    "asset_id": "fig_001",
                    "asset_uri": "images/page_001_figure_001.png",
                    "asset_type": "image",
                    "link_reason": "figure has caption immediately after image",
                }
            ],
            "metadata": {},
        }

        with self.assertRaises(AssertionError):
            assert_chunk_matches_golden_schema(self, shape, schema)

    def test_golden_schema_rejects_source_span_without_required_offsets(self):
        schema = json.loads(Path("tests/golden/chunk_schema.json").read_text(encoding="utf-8"))
        document = ParsedDocument(
            document_id="doc",
            source_path="doc.pdf",
            source_name="doc.pdf",
            page_total=1,
            blocks=[
                ParsedBlock(
                    block_id="b1",
                    document_id="doc",
                    page_no=1,
                    page_total=1,
                    sequence_no=1,
                    content_type="unknown",
                    text="1 DO OBJETO",
                    metadata={"label": "section_header"},
                ),
            ],
        )

        _, chunks, _ = run_pipeline(document, ParserPipelineConfig())
        shape = asdict(chunks[0])
        del shape["source_spans"][0]["block_start_char"]

        with self.assertRaises(AssertionError):
            assert_chunk_matches_golden_schema(self, shape, schema)

    def test_adapter_warnings_are_propagated_to_report(self):
        document = ParsedDocument(
            document_id="doc",
            source_path="doc.pdf",
            source_name="doc.pdf",
            page_total=1,
            blocks=[],
            metadata={"warnings": ["adapter warning"]},
        )

        _, _, report = run_pipeline(document, ParserPipelineConfig())

        self.assertIn("adapter warning", report.warnings)


if __name__ == "__main__":
    unittest.main()
