import unittest

from mini_rag.sparse import SparseRetriever, tokenize
from mini_rag.store import VectorItem


def record(
    chunk_id,
    content,
    *,
    source_chunk_id=None,
    source_spans=None,
    source_block_ids=None,
    page_start=None,
    page_end=None,
    section_path=None,
    related_assets=None,
):
    return {
        "chunk_id": chunk_id,
        "content": content,
        "metadata": {
            "source_chunk_id": source_chunk_id,
            "source_spans": list(source_spans or []),
            "source_block_ids": list(source_block_ids or []),
            "page_start": page_start,
            "page_end": page_end,
            "section_path": list(section_path or []),
            "related_assets": list(related_assets or []),
        },
    }


class SparseRetrieverTests(unittest.TestCase):
    def test_exact_term_ranks_matching_document_above_others(self):
        retriever = SparseRetriever(
            [
                record("c1", "objeto objeto da contratacao"),
                record("c2", "prazo de entrega"),
                record("c3", "valor estimado"),
            ]
        )

        results = retriever.search("objeto", top_k=3)

        self.assertEqual(results[0]["chunk"]["chunk_id"], "c1")
        self.assertGreater(results[0]["score"], 0.0)

    def test_top_k_is_respected(self):
        retriever = SparseRetriever(
            [
                record("c1", "prazo execucao"),
                record("c2", "prazo entrega"),
                record("c3", "prazo pagamento"),
            ]
        )

        results = retriever.search("prazo", top_k=2)

        self.assertEqual(len(results), 2)

    def test_query_without_match_returns_empty_list(self):
        retriever = SparseRetriever([record("c1", "objeto da contratacao")])

        results = retriever.search("inexistente", top_k=5)

        self.assertEqual(results, [])

    def test_ranking_is_deterministic(self):
        records = [
            record("c1", "prazo unico"),
            record("c2", "prazo unico"),
            record("c3", "prazo unico"),
        ]
        retriever = SparseRetriever(records)

        first_run = retriever.search("prazo", top_k=3)
        second_run = retriever.search("prazo", top_k=3)

        self.assertEqual(
            [result["chunk"]["chunk_id"] for result in first_run],
            ["c1", "c2", "c3"],
        )
        self.assertEqual(first_run, second_run)

    def test_auditable_metadata_is_preserved(self):
        spans = [{"block_id": "b1", "block_start_char": 0, "block_end_char": 5}]
        assets = [{"asset_id": "fig_001", "link_strategy": "caption_detected"}]
        retriever = SparseRetriever(
            [
                record(
                    "c1",
                    "objeto",
                    source_chunk_id="source-1",
                    source_spans=spans,
                    source_block_ids=["b1"],
                    page_start=2,
                    page_end=3,
                    section_path=["1 DO OBJETO"],
                    related_assets=assets,
                )
            ]
        )

        result = retriever.search("objeto", top_k=1)[0]

        self.assertEqual(result["chunk"]["chunk_id"], "c1")
        self.assertEqual(result["metadata"]["source_chunk_id"], "source-1")
        self.assertEqual(result["metadata"]["source_block_ids"], ["b1"])
        self.assertEqual(result["metadata"]["page_start"], 2)
        self.assertEqual(result["metadata"]["page_end"], 3)
        self.assertEqual(result["metadata"]["section_path"], ["1 DO OBJETO"])

    def test_source_spans_are_not_lost(self):
        spans = [{"block_id": "b1", "chunk_start_char": 0, "chunk_end_char": 6}]
        retriever = SparseRetriever([record("c1", "objeto", source_spans=spans)])

        result = retriever.search("objeto", top_k=1)[0]

        self.assertEqual(result["metadata"]["source_spans"], spans)

    def test_related_assets_are_not_lost(self):
        assets = [{"asset_id": "table_001", "link_reason": "same section"}]
        retriever = SparseRetriever([record("c1", "valor", related_assets=assets)])

        result = retriever.search("valor", top_k=1)[0]

        self.assertEqual(result["metadata"]["related_assets"], assets)

    def test_tokenization_handles_case_and_accents_predictably(self):
        self.assertEqual(
            tokenize("PRAZO, Contratação nº 007/2026."),
            ["prazo", "contratação", "nº", "007", "2026"],
        )

        retriever = SparseRetriever(
            [
                record("accented", "A contratação será auditável."),
                record("plain", "A contratacao sera auditavel."),
            ]
        )

        self.assertEqual(retriever.search("CONTRATAÇÃO", top_k=2)[0]["chunk"]["chunk_id"], "accented")
        self.assertEqual(retriever.search("contratacao", top_k=2)[0]["chunk"]["chunk_id"], "plain")

    def test_accepts_vector_store_items(self):
        retriever = SparseRetriever(
            [
                VectorItem(
                    chunk_id="c1",
                    content="prazo de execucao",
                    embedding=[0.0],
                    metadata={"source_chunk_id": "s1", "source_spans": [{"block_id": "b1"}]},
                )
            ]
        )

        result = retriever.search("prazo", top_k=1)[0]

        self.assertEqual(result["chunk"], {"chunk_id": "c1", "content": "prazo de execucao"})
        self.assertEqual(result["metadata"]["source_chunk_id"], "s1")
        self.assertEqual(result["metadata"]["source_spans"], [{"block_id": "b1"}])

    def test_top_level_audit_fields_are_preserved_for_raw_chunks(self):
        raw_chunk = {
            "chunk_id": "c1",
            "page_content": "valor estimado",
            "source_chunk_id": "source-raw",
            "source_spans": [{"block_id": "b1"}],
            "source_block_ids": ["b1"],
            "page_start": 4,
            "page_end": 4,
            "section_path": ["3 DO VALOR"],
            "related_assets": [{"asset_id": "fig_001"}],
        }
        retriever = SparseRetriever([raw_chunk])

        result = retriever.search("valor", top_k=1)[0]

        self.assertEqual(result["chunk"]["content"], "valor estimado")
        self.assertEqual(result["metadata"]["source_chunk_id"], "source-raw")
        self.assertEqual(result["metadata"]["source_spans"], [{"block_id": "b1"}])
        self.assertEqual(result["metadata"]["source_block_ids"], ["b1"])
        self.assertEqual(result["metadata"]["page_start"], 4)
        self.assertEqual(result["metadata"]["page_end"], 4)
        self.assertEqual(result["metadata"]["section_path"], ["3 DO VALOR"])
        self.assertEqual(result["metadata"]["related_assets"], [{"asset_id": "fig_001"}])


if __name__ == "__main__":
    unittest.main()
