import unittest

from mini_rag.citations import CitationBuilder


class MiniRagCitationTests(unittest.TestCase):
    def test_builds_auditable_citations_from_retrieved_chunks(self):
        results = [
            {
                "score": 0.93,
                "chunk": {"chunk_id": "c1", "content": "Texto."},
                "metadata": {
                    "source_chunk_id": "s1",
                    "page_start": 3,
                    "page_end": 4,
                    "section_path": ["1", "1.1"],
                    "source_block_ids": ["b1", "b2"],
                },
            }
        ]

        citations = CitationBuilder().build(results)

        self.assertEqual(
            citations,
            [
                {
                    "chunk_id": "c1",
                    "source_chunk_id": "s1",
                    "pages": [3, 4],
                    "page_start": 3,
                    "page_end": 4,
                    "section_path": ["1", "1.1"],
                    "source_block_ids": ["b1", "b2"],
                    "score": 0.93,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()

