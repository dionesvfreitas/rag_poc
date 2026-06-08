import unittest

from parser_core.application.normalizer import normalize_blocks
from parser_core.application.pipeline import ParserPipelineConfig
from parser_core.domain.models import BoundingBox, ParsedBlock


def block(text, page_no=1, page_total=1, label="text", bbox=None):
    return ParsedBlock(
        block_id=f"b{page_no}{text[:2]}",
        document_id="doc",
        page_no=page_no,
        page_total=page_total,
        sequence_no=page_no,
        content_type="unknown",
        text=text,
        bbox=bbox,
        metadata={"label": label},
    )


class NormalizerTests(unittest.TestCase):
    def test_whitespace_and_content_type_are_normalized(self):
        blocks, warnings = normalize_blocks(
            [block("A  proposta\n sera enviada")],
            ParserPipelineConfig(),
        )

        self.assertEqual(blocks[0].text, "A proposta sera enviada")
        self.assertEqual(blocks[0].content_type, "paragraph")
        self.assertEqual(warnings, [])

    def test_repeated_headers_are_marked_not_removed(self):
        blocks = [
            block("Repeated banner", page_no=1, page_total=4, bbox=BoundingBox(y0=0.05)),
            block("Repeated banner", page_no=2, page_total=4, bbox=BoundingBox(y0=0.05)),
            block("Repeated banner", page_no=3, page_total=4, bbox=BoundingBox(y0=0.05)),
            block("Body", page_no=4, page_total=4),
        ]

        normalized, warnings = normalize_blocks(blocks, ParserPipelineConfig())

        self.assertEqual(len(normalized), 4)
        self.assertTrue(normalized[0].metadata["repeated_header_footer_candidate"])
        self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()
