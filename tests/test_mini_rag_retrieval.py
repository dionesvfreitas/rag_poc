import unittest

from mini_rag.embeddings import EmbeddingProvider
from mini_rag.retrieval import Retriever
from mini_rag.store import LocalVectorStore


class QueryEmbeddingProvider(EmbeddingProvider):
    model_name = "fake"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class MiniRagRetrieverTests(unittest.TestCase):
    def test_search_respects_top_k_and_threshold(self):
        store = LocalVectorStore()
        store.add("c1", "Objeto", [1.0, 0.0])
        store.add("c2", "Objeto parecido", [0.9, 0.1])
        store.add("c3", "Irrelevante", [0.0, 1.0])
        retriever = Retriever(
            store,
            QueryEmbeddingProvider(),
            top_k=2,
            similarity_threshold=0.5,
        )

        results = retriever.search("objeto")

        self.assertEqual([result["chunk"]["chunk_id"] for result in results], ["c1", "c2"])


if __name__ == "__main__":
    unittest.main()

