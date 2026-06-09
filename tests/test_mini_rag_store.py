import tempfile
import unittest
from pathlib import Path

from mini_rag.store import LocalVectorStore, cosine_similarity


class MiniRagStoreTests(unittest.TestCase):
    def test_cosine_similarity_handles_matching_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_insert_save_load_and_search(self):
        store = LocalVectorStore(embedding_model="fake", source_path="rag_chunks.jsonl")
        store.add("c1", "Objeto", [1.0, 0.0], {"page_start": 2})
        store.add("c2", "Prazo", [0.0, 1.0], {"page_start": 5})

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "document_index.json"
            store.save(path)
            loaded = LocalVectorStore.load(path)

        results = loaded.search([1.0, 0.0], top_k=2)
        self.assertEqual([result["chunk"]["chunk_id"] for result in results], ["c1", "c2"])
        self.assertEqual(results[0]["metadata"]["page_start"], 2)


if __name__ == "__main__":
    unittest.main()

