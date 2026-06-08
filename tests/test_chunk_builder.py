import unittest

from parser_core.application.chunk_builder import ChunkBuilderConfig, build_chunks, split_large_chunk
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


def rebuild_chunk_from_spans(blocks_by_id, source_spans):
    parts = []
    for span in source_spans:
        if span.get("separator_before"):
            parts.append(span["separator_before"])
        source_text = blocks_by_id[span["block_id"]].text
        parts.append(source_text[span["block_start_char"] : span["block_end_char"]])
        if span.get("separator_after"):
            parts.append(span["separator_after"])
    return "".join(parts)


def assert_valid_source_spans(test_case, chunk, blocks_by_id):
    cursor = 0
    for span in chunk.source_spans:
        source_text = blocks_by_id[span["block_id"]].text
        separator_before = span.get("separator_before") or ""
        test_case.assertEqual(chunk.text[cursor : cursor + len(separator_before)], separator_before)
        cursor += len(separator_before)

        test_case.assertGreaterEqual(span["block_start_char"], 0)
        test_case.assertLessEqual(span["block_start_char"], span["block_end_char"])
        test_case.assertLessEqual(span["block_end_char"], len(source_text))
        test_case.assertEqual(span["start_char"], span["block_start_char"])
        test_case.assertEqual(span["end_char"], span["block_end_char"])

        test_case.assertEqual(span["chunk_start_char"], cursor)
        test_case.assertLessEqual(span["chunk_start_char"], span["chunk_end_char"])
        test_case.assertLessEqual(span["chunk_end_char"], len(chunk.text))
        test_case.assertEqual(
            chunk.text[span["chunk_start_char"] : span["chunk_end_char"]],
            source_text[span["block_start_char"] : span["block_end_char"]],
        )
        cursor = span["chunk_end_char"]

        separator_after = span.get("separator_after") or ""
        test_case.assertEqual(chunk.text[cursor : cursor + len(separator_after)], separator_after)
        cursor += len(separator_after)
    test_case.assertEqual(cursor, len(chunk.text))
    test_case.assertEqual(rebuild_chunk_from_spans(blocks_by_id, chunk.source_spans), chunk.text)


class ChunkBuilderTests(unittest.TestCase):
    def test_chunks_do_not_mix_sections_and_keep_source_blocks(self):
        blocks = [
            block("b1", "Body A", ["A"]),
            block("b2", "Body B", ["B"]),
        ]
        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(min_chunk_chars=200),
        )
        blocks_by_id = {item.block_id: item for item in blocks}

        self.assertEqual([chunk.section_path for chunk in chunks], [["A"], ["B"]])
        self.assertEqual(chunks[0].source_block_ids, ["b1"])
        self.assertEqual(chunks[0].source_spans[0]["block_id"], "b1")
        self.assertEqual(chunks[0].source_spans[0]["block_start_char"], 0)
        self.assertEqual(chunks[0].source_spans[0]["block_end_char"], 6)
        self.assertEqual(chunks[0].source_spans[0]["chunk_start_char"], 0)
        self.assertEqual(chunks[0].source_spans[0]["chunk_end_char"], 6)
        self.assertIsNone(chunks[0].source_spans[0]["separator_before"])
        assert_valid_source_spans(self, chunks[0], blocks_by_id)

    def test_tables_are_own_chunks(self):
        blocks = [
            block("b1", "Intro", ["A"]),
            block("b2", "| A | B |", ["A"], content_type="table"),
        ]
        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(),
        )
        blocks_by_id = {item.block_id: item for item in blocks}

        self.assertEqual(chunks[1].content_types, ["table"])
        self.assertEqual(chunks[1].source_block_ids, ["b2"])
        assert_valid_source_spans(self, chunks[1], blocks_by_id)

    def test_oversized_single_paragraph_is_split_under_max_chars(self):
        blocks = [block("b1", "x" * 95, ["A"])]
        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(max_chunk_chars=30, min_chunk_chars=1, merge_small_chunks=False),
        )
        blocks_by_id = {item.block_id: item for item in blocks}

        self.assertTrue(all(len(chunk.text) <= 30 for chunk in chunks))
        self.assertEqual([chunk.chunk_no for chunk in chunks], [1, 2, 3, 4])
        self.assertTrue(all(chunk.source_block_ids == ["b1"] for chunk in chunks))
        self.assertEqual(
            [
                [
                    {
                        "block_id": span["block_id"],
                        "start_char": span["start_char"],
                        "end_char": span["end_char"],
                        "split_index": span["split_index"],
                    }
                    for span in chunk.source_spans
                ]
                for chunk in chunks
            ],
            [
                [{"block_id": "b1", "start_char": 0, "end_char": 30, "split_index": 1}],
                [{"block_id": "b1", "start_char": 30, "end_char": 60, "split_index": 2}],
                [{"block_id": "b1", "start_char": 60, "end_char": 90, "split_index": 3}],
                [{"block_id": "b1", "start_char": 90, "end_char": 95, "split_index": 4}],
            ],
        )
        for chunk in chunks:
            assert_valid_source_spans(self, chunk, blocks_by_id)

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
        blocks = [
            block("b1", "Small A.", ["A"]),
            block("b2", "Small B.", ["A"]),
        ]
        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(min_chunk_chars=200),
        )
        blocks_by_id = {item.block_id: item for item in blocks}

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Small A.\n\nSmall B.")
        self.assertEqual(chunks[0].source_block_ids, ["b1", "b2"])
        self.assertEqual(
            [
                {
                    "block_id": span["block_id"],
                    "block_start_char": span["block_start_char"],
                    "block_end_char": span["block_end_char"],
                    "chunk_start_char": span["chunk_start_char"],
                    "chunk_end_char": span["chunk_end_char"],
                    "separator_before": span["separator_before"],
                    "separator_after": span["separator_after"],
                    "split_index": span["split_index"],
                    "start_char": span["start_char"],
                    "end_char": span["end_char"],
                }
                for span in chunks[0].source_spans
            ],
            [
                {
                    "block_id": "b1",
                    "block_start_char": 0,
                    "block_end_char": 8,
                    "chunk_start_char": 0,
                    "chunk_end_char": 8,
                    "separator_before": None,
                    "separator_after": None,
                    "split_index": 0,
                    "start_char": 0,
                    "end_char": 8,
                },
                {
                    "block_id": "b2",
                    "block_start_char": 0,
                    "block_end_char": 8,
                    "chunk_start_char": 10,
                    "chunk_end_char": 18,
                    "separator_before": "\n\n",
                    "separator_after": None,
                    "split_index": 0,
                    "start_char": 0,
                    "end_char": 8,
                },
            ],
        )
        assert_valid_source_spans(self, chunks[0], blocks_by_id)

    def test_split_chunk_spans_can_reconstruct_chunk_text_from_original_slices(self):
        text = "First sentence. Second sentence. Third sentence."
        blocks = [block("b1", text, ["A"])]
        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(max_chunk_chars=32, min_chunk_chars=1, merge_small_chunks=False),
        )
        blocks_by_id = {item.block_id: item for item in blocks}

        for chunk in chunks:
            assert_valid_source_spans(self, chunk, blocks_by_id)
        self.assertEqual(
            [span["split_index"] for chunk in chunks for span in chunk.source_spans],
            [1, 2],
        )

    def test_multi_block_chunk_spans_reconstruct_with_separators(self):
        blocks = [
            block("b1", "First paragraph.", ["A"]),
            block("b2", "Second paragraph.", ["A"]),
            block("b3", "Third paragraph.", ["A"]),
        ]
        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(max_chunk_chars=36, min_chunk_chars=1, merge_small_chunks=False),
        )
        blocks_by_id = {item.block_id: item for item in blocks}

        self.assertEqual([chunk.text for chunk in chunks], ["First paragraph.\n\nSecond paragraph.", "Third paragraph."])
        self.assertEqual(chunks[0].source_block_ids, ["b1", "b2"])
        self.assertEqual([span["split_index"] for span in chunks[0].source_spans], [0, 0])
        self.assertEqual(chunks[0].source_spans[1]["separator_before"], "\n\n")
        for chunk in chunks:
            assert_valid_source_spans(self, chunk, blocks_by_id)

    def test_split_multi_block_chunk_keeps_only_spanned_source_block_ids(self):
        blocks = [
            block("b1", "First paragraph.", ["A"]),
            block("b2", "Second paragraph.", ["A"]),
            block("b3", "Third paragraph.", ["A"]),
        ]
        chunks = build_chunks(
            blocks,
            "doc",
            ChunkBuilderConfig(max_chunk_chars=100, min_chunk_chars=1, merge_small_chunks=False),
        )
        split_chunks = split_large_chunk(
            chunks[0],
            ChunkBuilderConfig(max_chunk_chars=36, min_chunk_chars=1, merge_small_chunks=False),
        )
        blocks_by_id = {item.block_id: item for item in blocks}

        self.assertEqual([chunk.text for chunk in split_chunks], ["First paragraph.\n\nSecond paragraph.", "Third paragraph."])
        self.assertEqual(split_chunks[0].source_block_ids, ["b1", "b2"])
        self.assertEqual(split_chunks[1].source_block_ids, ["b3"])
        self.assertEqual(
            split_chunks[1].source_block_ids,
            [span["block_id"] for span in split_chunks[1].source_spans],
        )
        for chunk in split_chunks:
            assert_valid_source_spans(self, chunk, blocks_by_id)

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
