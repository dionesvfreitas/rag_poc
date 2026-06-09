import json
import tempfile
import unittest
from pathlib import Path

from mini_rag.embeddings import EmbeddingProvider, SentenceTransformersEmbeddingProvider
from mini_rag.indexing import build_document_index
from settings import Settings


class FakeModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        return [[float(len(text)), 1.0] for text in texts]


class FakeEmbeddingProvider(EmbeddingProvider):
    model_name = "fake-model"

    def embed(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class MiniRagEmbeddingTests(unittest.TestCase):
    def test_sentence_transformers_provider_generates_embedding_from_injected_model(self):
        provider = SentenceTransformersEmbeddingProvider("fake-model", model=FakeModel())

        embeddings = provider.embed(["abc"])

        self.assertEqual(embeddings, [[3.0, 1.0]])

    def test_embedding_is_persisted_in_document_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "rag_chunks.jsonl"
            output_path = Path(tmp) / "document_index.json"
            input_path.write_text(
                json.dumps(
                    {
                        "chunk_type": "child",
                        "chunk_id": "c1",
                        "content": "Objeto do edital.",
                        "source_chunk_id": "s1",
                        "page_start": 2,
                        "page_end": 2,
                        "section_path": ["1 DO OBJETO"],
                        "source_block_ids": ["b1"],
                        "metadata": {},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            build_document_index(
                input_path,
                output_path,
                embedding_provider=FakeEmbeddingProvider(),
                settings=Settings(index_path=str(output_path)),
                input_type="rag_chunks",
            )

            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(data["embedding_model"], "fake-model")
            self.assertEqual(data["items"][0]["embedding"], [17.0, 1.0])


if __name__ == "__main__":
    unittest.main()

