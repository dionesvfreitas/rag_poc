import unittest

from parser_core.application.chunk_builder import ChunkBuilderConfig, build_chunks
from parser_core.domain.models import ParsedBlock


def block(block_id, text, section_path, content_type="paragraph", page_no=1):
    return ParsedBlock(
        block_id=block_id,
        document_id="doc",
        page_no=page_no,
        page_total=2,
        sequence_no=int(block_id[1:]),
        content_type=content_type,
        text=text,
        section_title=section_path[-1] if section_path else None,
        section_path=section_path,
        metadata={"label": "table" if content_type == "table" else "text"},
    )


class ChunkBuilderTests(unittest.TestCase):
    def test_chunks_do_not_mix_sections_and_keep_source_blocks(self):
        chunks = build_chunks(
            [
                block("b1", "Body A", ["A"]),
                block("b2", "Body B", ["B"]),
            ],
            "doc",
            ChunkBuilderConfig(min_chunk_chars=200),
        )

        self.assertEqual([chunk.section_path for chunk in chunks], [["A"], ["B"]])
        self.assertEqual(chunks[0].source_block_ids, ["b1"])
        self.assertEqual(
            chunks[0].source_spans,
            [{"block_id": "b1", "start_char": 0, "end_char": 6, "split_index": 0}],
        )

    def test_tables_are_own_chunks(self):
        chunks = build_chunks(
            [
                block("b1", "Intro", ["A"]),
                block("b2", "| A | B |", ["A"], content_type="table"),
            ],
            "doc",
            ChunkBuilderConfig(),
        )

        self.assertEqual(chunks[1].content_types, ["table"])

    def test_oversized_single_paragraph_is_split_under_max_chars(self):
        chunks = build_chunks(
            [block("b1", "x" * 95, ["A"])],
            "doc",
            ChunkBuilderConfig(max_chunk_chars=30, min_chunk_chars=1, merge_small_chunks=False),
        )

        self.assertTrue(all(len(chunk.text) <= 30 for chunk in chunks))
        self.assertEqual([chunk.chunk_no for chunk in chunks], [1, 2, 3, 4])
        self.assertTrue(all(chunk.source_block_ids == ["b1"] for chunk in chunks))
        self.assertEqual(
            [chunk.source_spans for chunk in chunks],
            [
                [{"block_id": "b1", "start_char": 0, "end_char": 30, "split_index": 1}],
                [{"block_id": "b1", "start_char": 30, "end_char": 60, "split_index": 2}],
                [{"block_id": "b1", "start_char": 60, "end_char": 90, "split_index": 3}],
                [{"block_id": "b1", "start_char": 90, "end_char": 95, "split_index": 4}],
            ],
        )

    def test_final_chunk_numbers_are_sequential_after_split(self):
        chunks = build_chunks(
            [
                block("b1", "x" * 70, ["A"]),
                block("b2", "Second chunk.", ["A"]),
            ],
            "doc",
            ChunkBuilderConfig(max_chunk_chars=25, min_chunk_chars=1, merge_small_chunks=False),
        )

        self.assertEqual([chunk.chunk_no for chunk in chunks], list(range(1, len(chunks) + 1)))

    def test_merged_chunks_keep_source_spans_for_each_block(self):
        chunks = build_chunks(
            [
                block("b1", "Small A.", ["A"]),
                block("b2", "Small B.", ["A"]),
            ],
            "doc",
            ChunkBuilderConfig(min_chunk_chars=200),
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].source_spans,
            [
                {"block_id": "b1", "start_char": 0, "end_char": 8, "split_index": 0},
                {"block_id": "b2", "start_char": 0, "end_char": 8, "split_index": 0},
            ],
        )

    def test_split_chunk_spans_can_reconstruct_chunk_text_from_original_slices(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = build_chunks(
            [block("b1", text, ["A"])],
            "doc",
            ChunkBuilderConfig(max_chunk_chars=32, min_chunk_chars=1, merge_small_chunks=False),
        )

        for chunk in chunks:
            reconstructed = "".join(
                text[span["start_char"] : span["end_char"]] for span in chunk.source_spans
            )
            self.assertEqual(reconstructed, chunk.text)
        self.assertEqual(
            [span["split_index"] for chunk in chunks for span in chunk.source_spans],
            [1, 2],
        )

    def test_split_prefers_sentence_boundaries_before_character_limit(self):
        text = "First sentence is complete. Second sentence is complete."
        chunks = build_chunks(
            [block("b1", text, ["A"])],
            "doc",
            ChunkBuilderConfig(max_chunk_chars=30, min_chunk_chars=1, merge_small_chunks=False),
        )

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["First sentence is complete.", "Second sentence is complete."],
        )
        self.assertFalse(chunks[1].metadata["starts_mid_sentence"])
        self.assertEqual(chunks[1].metadata["split_strategy"], "sentence")
        self.assertFalse(chunks[1].metadata["hard_split"])

    def test_character_split_keeps_leading_punctuation_with_previous_piece(self):
        text = "ABCDEFGHIJ , continuation"
        chunks = build_chunks(
            [block("b1", text, ["A"])],
            "doc",
            ChunkBuilderConfig(max_chunk_chars=11, min_chunk_chars=1, merge_small_chunks=False),
        )

        self.assertEqual(chunks[0].text, "ABCDEFGHIJ ,")
        self.assertTrue(all(not chunk.text.startswith(",") for chunk in chunks))

    def test_unavoidable_character_split_marks_mid_sentence(self):
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = build_chunks(
            [block("b1", text, ["A"])],
            "doc",
            ChunkBuilderConfig(max_chunk_chars=10, min_chunk_chars=1, merge_small_chunks=False),
        )

        self.assertEqual([chunk.text for chunk in chunks], ["ABCDEFGHIJ", "KLMNOPQRST", "UVWXYZ"])
        self.assertFalse(chunks[0].metadata["starts_mid_sentence"])
        self.assertTrue(chunks[1].metadata["starts_mid_sentence"])
        self.assertTrue(chunks[2].metadata["starts_mid_sentence"])
        self.assertTrue(chunks[1].metadata["hard_split"])


if __name__ == "__main__":
    unittest.main()
