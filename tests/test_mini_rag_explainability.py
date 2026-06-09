import unittest

from mini_rag.indexing import audit_metadata
from query import format_results


class MiniRagExplainabilityTests(unittest.TestCase):
    def test_audit_metadata_preserves_required_fields(self):
        chunk = {
            "chunk_type": "child",
            "chunk_id": "c1",
            "source_chunk_id": "s1",
            "content": "Qual o objeto?",
            "page_start": 2,
            "page_end": 2,
            "section_path": ["1 DO OBJETO"],
            "source_block_ids": ["b1"],
            "source_spans": [{"block_id": "b1"}],
            "related_assets": [{"asset_id": "a1"}],
            "content_types": ["paragraph"],
            "metadata": {"clause_path": ["1", "1.1"]},
        }

        metadata = audit_metadata(chunk)

        self.assertEqual(metadata["source_chunk_id"], "s1")
        self.assertEqual(metadata["page_start"], 2)
        self.assertEqual(metadata["page_end"], 2)
        self.assertEqual(metadata["section_path"], ["1 DO OBJETO"])
        self.assertEqual(metadata["source_block_ids"], ["b1"])
        self.assertEqual(metadata["source_spans"], [{"block_id": "b1"}])
        self.assertEqual(metadata["related_assets"], [{"asset_id": "a1"}])

    def test_query_output_includes_explainability_fields(self):
        output = format_results(
            [
                {
                    "score": 0.93,
                    "chunk": {"chunk_id": "c1", "content": "Objeto do edital."},
                    "metadata": {
                        "source_chunk_id": "s1",
                        "page_start": 2,
                        "page_end": 2,
                        "section_path": ["1 DO OBJETO"],
                        "source_block_ids": ["b1"],
                    },
                }
            ]
        )

        self.assertIn("score=0.9300", output)
        self.assertIn("chunk_id: c1", output)
        self.assertIn("source_chunk_id: s1", output)
        self.assertIn("page: 2", output)
        self.assertIn("section: 1 DO OBJETO", output)
        self.assertIn("source_block_ids: b1", output)


if __name__ == "__main__":
    unittest.main()
