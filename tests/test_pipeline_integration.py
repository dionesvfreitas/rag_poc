import unittest
import json
from dataclasses import asdict
from pathlib import Path

from parser_core.application.pipeline import ParserPipelineConfig, run_pipeline
from parser_core.domain.models import ParsedBlock, ParsedDocument


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

        self.assertTrue(all(field in shape for field in schema["required_fields"]))

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
