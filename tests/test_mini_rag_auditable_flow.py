import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_rag import indexing
from mini_rag.citations import CitationBuilder
from mini_rag.embeddings import EmbeddingProvider
from mini_rag.indexing import build_document_index
from mini_rag.retrieval import Retriever
from mini_rag.store import LocalVectorStore
from settings import Settings


class KeywordEmbeddingProvider(EmbeddingProvider):
    model_name = "fake-keyword-embedding"

    def embed(self, texts):
        vectors = []
        for text in texts:
            if "objeto" in text.lower():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def single_block_span(block_id, text, page_no):
    return {
        "block_id": block_id,
        "block_start_char": 0,
        "block_end_char": len(text),
        "chunk_start_char": 0,
        "chunk_end_char": len(text),
        "separator_before": None,
        "separator_after": None,
        "split_index": 0,
        "page_no": page_no,
    }


def parsed_section(chunk_id, text, *, block_id, page_no, section_title, section_path):
    return {
        "document_id": "edital.pdf",
        "chunk_id": chunk_id,
        "parent_chunk_id": "parser-parent",
        "section_title": section_title,
        "subsection_title": None,
        "section_path": section_path,
        "clause_number": None,
        "page_no": page_no,
        "page_start": page_no,
        "page_end": page_no,
        "page_total": 4,
        "content_type": "paragraph",
        "content_types": ["paragraph"],
        "page_content": text,
        "source_block_ids": [block_id],
        "source_spans": [single_block_span(block_id, text, page_no)],
        "document_region": "body",
        "related_assets": [],
        "metadata": {"clause_path": []},
    }


class MiniRagAuditableFlowTests(unittest.TestCase):
    def test_parsed_sections_flow_builds_loads_retrieves_and_cites_locally(self):
        object_text = "O objeto da contratação é a prestação de serviços auditáveis."
        prazo_text = "O prazo de execução será de 30 dias corridos."

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "parsed_sections.jsonl"
            output_path = tmp_path / "document_index.json"
            write_jsonl(
                input_path,
                [
                    parsed_section(
                        "parsed-objeto",
                        object_text,
                        block_id="block-objeto",
                        page_no=2,
                        section_title="1 DO OBJETO",
                        section_path=["1 DO OBJETO"],
                    ),
                    parsed_section(
                        "parsed-prazo",
                        prazo_text,
                        block_id="block-prazo",
                        page_no=3,
                        section_title="2 DO PRAZO",
                        section_path=["2 DO PRAZO"],
                    ),
                ],
            )

            with patch(
                "mini_rag.indexing.build_hierarchical_chunks",
                wraps=indexing.build_hierarchical_chunks,
            ) as build_chunks:
                build_document_index(
                    input_path,
                    output_path,
                    embedding_provider=KeywordEmbeddingProvider(),
                    settings=Settings(index_path=str(output_path)),
                    input_type="parsed_sections",
                )

            self.assertTrue(build_chunks.called)
            self.assertTrue(output_path.exists())
            index_data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(index_data["items"]), 1)

            object_item = next(
                item for item in index_data["items"] if item["metadata"]["source_chunk_id"] == "parsed-objeto"
            )
            for field in ["chunk_id", "content", "embedding", "metadata"]:
                self.assertIn(field, object_item)
            self.assertEqual(object_item["content"], object_text)
            self.assertEqual(object_item["embedding"], [1.0, 0.0])

            metadata = object_item["metadata"]
            self.assertEqual(metadata["source_spans"], [single_block_span("block-objeto", object_text, 2)])
            self.assertEqual(metadata["source_block_ids"], ["block-objeto"])
            self.assertEqual(metadata["page_start"], 2)
            self.assertEqual(metadata["page_end"], 2)
            self.assertEqual(metadata["section_path"], ["1 DO OBJETO"])
            self.assertEqual(metadata["section_title"], "1 DO OBJETO")
            self.assertEqual(metadata["document_region"], "body")
            self.assertEqual(metadata["related_assets"], [])

            loaded_store = LocalVectorStore.load(output_path)
            retriever = Retriever(
                loaded_store,
                KeywordEmbeddingProvider(),
                top_k=2,
                similarity_threshold=None,
            )
            results = retriever.search("Qual o objeto?", top_k=2)

            self.assertGreaterEqual(len(results), 1)
            self.assertEqual(results[0]["metadata"]["source_chunk_id"], "parsed-objeto")
            self.assertIn("objeto", results[0]["chunk"]["content"].lower())

            citations = CitationBuilder().build(results)

            self.assertGreaterEqual(len(citations), 1)
            self.assertEqual(citations[0]["chunk_id"], results[0]["chunk"]["chunk_id"])
            self.assertEqual(citations[0]["page_start"], 2)
            self.assertEqual(citations[0]["page_end"], 2)
            self.assertEqual(citations[0]["pages"], [2])
            self.assertEqual(citations[0]["section_path"], ["1 DO OBJETO"])
            self.assertEqual(citations[0]["source_block_ids"], ["block-objeto"])
            self.assertEqual(citations[0]["score"], results[0]["score"])

    def test_search_tie_keeps_original_insertion_order(self):
        store = LocalVectorStore()
        store.add("chunk-a", "Objeto A", [1.0, 0.0])
        store.add("chunk-b", "Objeto B", [1.0, 0.0])

        results = store.search([1.0, 0.0], top_k=2)

        self.assertEqual([result["chunk"]["chunk_id"] for result in results], ["chunk-a", "chunk-b"])


if __name__ == "__main__":
    unittest.main()
