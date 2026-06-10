import unittest

from mini_rag.hybrid import HybridRetriever


class FakeRetriever:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def search(self, query, top_k=None):
        self.calls.append((query, top_k))
        if top_k is None:
            return list(self.results)
        return list(self.results[:top_k])


def result(
    chunk_id,
    score,
    *,
    content=None,
    source_chunk_id=None,
    source_spans=None,
    source_block_ids=None,
    related_assets=None,
    citations=None,
    page_start=None,
    page_end=None,
    section_path=None,
):
    metadata = {
        "source_chunk_id": source_chunk_id,
        "source_spans": list(source_spans or []),
        "source_block_ids": list(source_block_ids or []),
        "related_assets": list(related_assets or []),
        "citations": list(citations or []),
        "page_start": page_start,
        "page_end": page_end,
        "section_path": list(section_path or []),
    }
    return {
        "score": score,
        "chunk": {"chunk_id": chunk_id, "content": content or f"content {chunk_id}"},
        "metadata": metadata,
    }


class HybridRetrieverTests(unittest.TestCase):
    def test_rrf_fusion_is_deterministic(self):
        retriever = HybridRetriever(
            FakeRetriever([result("c1", 0.9), result("c2", 0.8)]),
            FakeRetriever([result("c2", 7.0), result("c3", 6.0)]),
            fusion_strategy="rrf",
            rrf_k=60,
        )

        first_run = retriever.search("objeto", top_k=3)
        second_run = retriever.search("objeto", top_k=3)

        self.assertEqual([item["chunk_id"] for item in first_run], ["c2", "c1", "c3"])
        self.assertEqual(first_run, second_run)
        self.assertAlmostEqual(first_run[0]["score"], (1 / 62) + (1 / 61))

    def test_weighted_fusion_is_deterministic_without_implicit_normalization(self):
        retriever = HybridRetriever(
            FakeRetriever([result("c1", 0.2), result("c2", 0.9)]),
            FakeRetriever([result("c1", 10.0), result("c3", 0.1)]),
            fusion_strategy="weighted",
            dense_weight=1.0,
            sparse_weight=0.1,
        )

        results = retriever.search("valor", top_k=3)

        self.assertEqual([item["chunk_id"] for item in results], ["c1", "c2", "c3"])
        self.assertAlmostEqual(results[0]["score"], 1.2)
        self.assertAlmostEqual(results[0]["diagnostics"]["weighted_score"], 1.2)

    def test_deduplicates_by_chunk_id(self):
        retriever = HybridRetriever(
            FakeRetriever([result("same", 0.9, source_chunk_id="dense-source")]),
            FakeRetriever([result("same", 4.0, source_chunk_id="sparse-source")]),
        )

        results = retriever.search("prazo", top_k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "same")
        self.assertEqual(results[0]["diagnostics"]["dense_score"], 0.9)
        self.assertEqual(results[0]["diagnostics"]["sparse_score"], 4.0)

    def test_ties_keep_deterministic_input_order(self):
        retriever = HybridRetriever(
            FakeRetriever([result("c1", 0.5), result("c2", 0.5)]),
            FakeRetriever([]),
            fusion_strategy="weighted",
            dense_weight=1.0,
            sparse_weight=0.0,
        )

        results = retriever.search("empate", top_k=2)

        self.assertEqual([item["chunk_id"] for item in results], ["c1", "c2"])

    def test_fallback_when_dense_returns_empty(self):
        retriever = HybridRetriever(
            FakeRetriever([]),
            FakeRetriever([result("sparse-only", 2.0)]),
        )

        results = retriever.search("lexical", top_k=2)

        self.assertEqual([item["chunk_id"] for item in results], ["sparse-only"])
        self.assertIsNone(results[0]["diagnostics"]["dense_score"])
        self.assertEqual(results[0]["diagnostics"]["sparse_score"], 2.0)

    def test_fallback_when_sparse_returns_empty(self):
        retriever = HybridRetriever(
            FakeRetriever([result("dense-only", 0.7)]),
            FakeRetriever([]),
        )

        results = retriever.search("semantic", top_k=2)

        self.assertEqual([item["chunk_id"] for item in results], ["dense-only"])
        self.assertEqual(results[0]["diagnostics"]["dense_score"], 0.7)
        self.assertIsNone(results[0]["diagnostics"]["sparse_score"])

    def test_fallback_when_both_retrievers_return_empty(self):
        retriever = HybridRetriever(FakeRetriever([]), FakeRetriever([]))

        self.assertEqual(retriever.search("sem evidencias", top_k=3), [])

    def test_source_spans_are_preserved(self):
        spans = [{"block_id": "b1", "chunk_start_char": 0, "chunk_end_char": 5}]
        retriever = HybridRetriever(
            FakeRetriever([result("c1", 0.9, source_spans=spans)]),
            FakeRetriever([]),
        )

        hybrid_result = retriever.search("objeto", top_k=1)[0]

        self.assertEqual(hybrid_result["metadata"]["source_spans"], spans)

    def test_related_assets_are_preserved(self):
        assets = [{"asset_id": "fig_001", "link_strategy": "caption_detected"}]
        retriever = HybridRetriever(
            FakeRetriever([]),
            FakeRetriever([result("c1", 2.0, related_assets=assets)]),
        )

        hybrid_result = retriever.search("figura", top_k=1)[0]

        self.assertEqual(hybrid_result["metadata"]["related_assets"], assets)

    def test_citations_are_preserved(self):
        citations = [{"chunk_id": "c1", "page_start": 2, "page_end": 2}]
        retriever = HybridRetriever(
            FakeRetriever([result("c1", 0.9, citations=citations)]),
            FakeRetriever([]),
        )

        hybrid_result = retriever.search("citacao", top_k=1)[0]

        self.assertEqual(hybrid_result["metadata"]["citations"], citations)

    def test_page_start_and_page_end_are_preserved(self):
        retriever = HybridRetriever(
            FakeRetriever([result("c1", 0.9, page_start=4, page_end=5)]),
            FakeRetriever([]),
        )

        hybrid_result = retriever.search("paginas", top_k=1)[0]

        self.assertEqual(hybrid_result["metadata"]["page_start"], 4)
        self.assertEqual(hybrid_result["metadata"]["page_end"], 5)

    def test_source_block_ids_and_section_path_are_preserved(self):
        retriever = HybridRetriever(
            FakeRetriever([]),
            FakeRetriever(
                [
                    result(
                        "c1",
                        2.0,
                        source_block_ids=["b1", "b2"],
                        section_path=["1 DO OBJETO", "1.1 ESCOPO"],
                    )
                ]
            ),
        )

        hybrid_result = retriever.search("secao", top_k=1)[0]

        self.assertEqual(hybrid_result["metadata"]["source_block_ids"], ["b1", "b2"])
        self.assertEqual(
            hybrid_result["metadata"]["section_path"],
            ["1 DO OBJETO", "1.1 ESCOPO"],
        )

    def test_source_chunk_id_is_preserved(self):
        retriever = HybridRetriever(
            FakeRetriever([result("c1", 0.9, source_chunk_id="source-1")]),
            FakeRetriever([]),
        )

        hybrid_result = retriever.search("origem", top_k=1)[0]

        self.assertEqual(hybrid_result["source_chunk_id"], "source-1")
        self.assertEqual(hybrid_result["metadata"]["source_chunk_id"], "source-1")

    def test_diagnostics_are_present(self):
        retriever = HybridRetriever(
            FakeRetriever([result("c1", 0.9)]),
            FakeRetriever([result("c1", 3.0)]),
            rrf_k=60,
        )

        diagnostics = retriever.search("diagnostico", top_k=1)[0]["diagnostics"]

        self.assertEqual(
            set(diagnostics),
            {
                "dense_score",
                "sparse_score",
                "rrf_score",
                "weighted_score",
                "fusion_strategy",
            },
        )
        self.assertEqual(diagnostics["dense_score"], 0.9)
        self.assertEqual(diagnostics["sparse_score"], 3.0)

    def test_fusion_strategy_is_reported(self):
        retriever = HybridRetriever(
            FakeRetriever([result("c1", 0.9)]),
            FakeRetriever([]),
            fusion_strategy="weighted",
        )

        hybrid_result = retriever.search("estrategia", top_k=1)[0]

        self.assertEqual(hybrid_result["diagnostics"]["fusion_strategy"], "weighted")


if __name__ == "__main__":
    unittest.main()
