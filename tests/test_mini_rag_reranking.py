import unittest

from mini_rag.hybrid import HybridRetriever
from mini_rag.reranking import CrossEncoderReranker, FakeReranker, IdentityReranker


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
    source_spans=None,
    related_assets=None,
    citations=None,
    diagnostics=None,
    metadata=None,
):
    result_metadata = dict(metadata or {})
    result_metadata.setdefault("source_spans", list(source_spans or []))
    result_metadata.setdefault("related_assets", list(related_assets or []))
    result_metadata.setdefault("citations", list(citations or []))
    return {
        "chunk_id": chunk_id,
        "score": score,
        "chunk": {"chunk_id": chunk_id, "content": content or f"content {chunk_id}"},
        "metadata": result_metadata,
        "diagnostics": dict(diagnostics or {}),
    }


class RerankingTests(unittest.TestCase):
    def test_identity_reranker_preserves_order(self):
        results = [result("c1", 0.3), result("c2", 0.9)]

        reranked = IdentityReranker().rerank("query", results)

        self.assertEqual([item["chunk_id"] for item in reranked], ["c1", "c2"])
        self.assertEqual(reranked[0]["diagnostics"]["reranker"], "identity")

    def test_fake_reranker_reorders_deterministically_by_explicit_score(self):
        results = [result("c1", 0.9), result("c2", 0.1), result("c3", 0.5)]
        reranker = FakeReranker(scores_by_chunk_id={"c2": 10.0, "c3": 3.0, "c1": 1.0})

        reranked = reranker.rerank("query", results)

        self.assertEqual([item["chunk_id"] for item in reranked], ["c2", "c3", "c1"])
        self.assertEqual([item["diagnostics"]["rerank_score"] for item in reranked], [10.0, 3.0, 1.0])

    def test_fake_reranker_can_use_injected_score_key(self):
        results = [
            result("c1", 0.9, metadata={"fake_rerank_score": 2.0}),
            result("c2", 0.1, diagnostics={"fake_rerank_score": 5.0}),
        ]

        reranked = FakeReranker().rerank("query", results)

        self.assertEqual([item["chunk_id"] for item in reranked], ["c2", "c1"])

    def test_top_k_is_respected(self):
        results = [result("c1", 0.1), result("c2", 0.2), result("c3", 0.3)]
        reranker = FakeReranker(scores_by_chunk_id={"c1": 1, "c2": 2, "c3": 3})

        reranked = reranker.rerank("query", results, top_k=2)

        self.assertEqual([item["chunk_id"] for item in reranked], ["c3", "c2"])

    def test_metadata_is_preserved(self):
        metadata = {"source_chunk_id": "source-1", "page_start": 2, "page_end": 3}
        reranked = FakeReranker(scores_by_chunk_id={"c1": 1}).rerank(
            "query",
            [result("c1", 0.5, metadata=metadata)],
        )

        self.assertEqual(reranked[0]["metadata"]["source_chunk_id"], "source-1")
        self.assertEqual(reranked[0]["metadata"]["page_start"], 2)
        self.assertEqual(reranked[0]["metadata"]["page_end"], 3)

    def test_source_spans_are_preserved(self):
        spans = [{"block_id": "b1", "chunk_start_char": 0, "chunk_end_char": 10}]

        reranked = FakeReranker(scores_by_chunk_id={"c1": 1}).rerank(
            "query",
            [result("c1", 0.5, source_spans=spans)],
        )

        self.assertEqual(reranked[0]["metadata"]["source_spans"], spans)

    def test_related_assets_are_preserved(self):
        assets = [{"asset_id": "fig_001", "link_strategy": "caption_detected"}]

        reranked = FakeReranker(scores_by_chunk_id={"c1": 1}).rerank(
            "query",
            [result("c1", 0.5, related_assets=assets)],
        )

        self.assertEqual(reranked[0]["metadata"]["related_assets"], assets)

    def test_citations_are_preserved(self):
        citations = [{"chunk_id": "c1", "page_start": 4, "page_end": 4}]

        reranked = FakeReranker(scores_by_chunk_id={"c1": 1}).rerank(
            "query",
            [result("c1", 0.5, citations=citations)],
        )

        self.assertEqual(reranked[0]["metadata"]["citations"], citations)

    def test_rerank_score_is_added(self):
        reranked = FakeReranker(scores_by_chunk_id={"c1": 7.5}).rerank(
            "query",
            [result("c1", 0.5)],
        )

        self.assertEqual(reranked[0]["diagnostics"]["rerank_score"], 7.5)
        self.assertEqual(reranked[0]["diagnostics"]["reranker"], "fake")

    def test_original_score_and_existing_diagnostics_are_not_lost(self):
        original = result(
            "c1",
            0.42,
            diagnostics={
                "dense_score": 0.4,
                "sparse_score": 2.0,
                "hybrid_score": 0.42,
                "rrf_score": 0.03,
                "weighted_score": 1.2,
                "fusion_strategy": "weighted",
            },
        )

        reranked = FakeReranker(scores_by_chunk_id={"c1": 9.0}).rerank("query", [original])
        diagnostics = reranked[0]["diagnostics"]

        self.assertEqual(reranked[0]["score"], 0.42)
        self.assertEqual(diagnostics["original_score"], 0.42)
        self.assertEqual(diagnostics["dense_score"], 0.4)
        self.assertEqual(diagnostics["sparse_score"], 2.0)
        self.assertEqual(diagnostics["hybrid_score"], 0.42)
        self.assertEqual(diagnostics["rrf_score"], 0.03)
        self.assertEqual(diagnostics["weighted_score"], 1.2)
        self.assertEqual(diagnostics["fusion_strategy"], "weighted")

    def test_cross_encoder_reranker_does_not_load_model_when_instantiated(self):
        reranker = CrossEncoderReranker(model_name="missing-local-model")

        self.assertEqual(reranker.model_name, "missing-local-model")
        self.assertIsNone(reranker._model)

    def test_hybrid_retriever_applies_optional_reranking(self):
        reranker = FakeReranker(scores_by_chunk_id={"c1": 1.0, "c2": 10.0})
        retriever = HybridRetriever(
            FakeRetriever([result("c1", 0.9), result("c2", 0.8)]),
            FakeRetriever([]),
            fusion_strategy="weighted",
            dense_weight=1.0,
            sparse_weight=0.0,
            reranker=reranker,
        )

        reranked = retriever.search("query", top_k=1)

        self.assertEqual([item["chunk_id"] for item in reranked], ["c2"])
        self.assertEqual(reranked[0]["diagnostics"]["rerank_score"], 10.0)
        self.assertEqual(reranked[0]["diagnostics"]["dense_score"], 0.8)

    def test_hybrid_retriever_without_reranker_keeps_previous_order(self):
        retriever = HybridRetriever(
            FakeRetriever([result("c1", 0.9), result("c2", 0.8)]),
            FakeRetriever([]),
            fusion_strategy="weighted",
            dense_weight=1.0,
            sparse_weight=0.0,
        )

        results = retriever.search("query", top_k=1)

        self.assertEqual([item["chunk_id"] for item in results], ["c1"])
        self.assertNotIn("rerank_score", results[0]["diagnostics"])


if __name__ == "__main__":
    unittest.main()
