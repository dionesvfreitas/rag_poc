from dataclasses import dataclass
from typing import Any, Protocol

from mini_rag.reranking import Reranker


FUSION_RRF = "rrf"
FUSION_WEIGHTED = "weighted"
SUPPORTED_FUSION_STRATEGIES = (FUSION_RRF, FUSION_WEIGHTED)
AUDIT_LIST_FIELDS = (
    "source_spans",
    "source_block_ids",
    "section_path",
    "related_assets",
    "citations",
)
AUDIT_SCALAR_FIELDS = (
    "source_chunk_id",
    "page_start",
    "page_end",
)


class SearchRetriever(Protocol):
    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        ...


@dataclass
class _Candidate:
    chunk_id: str
    first_seen_order: int
    dense_result: dict[str, Any] | None = None
    sparse_result: dict[str, Any] | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None


class HybridRetriever:
    """Experimental retriever that fuses injected dense and sparse retrievers.

    The weighted strategy uses raw retriever scores directly:

        hybrid_score = dense_weight * dense_score + sparse_weight * sparse_score

    Missing retriever scores contribute zero to the weighted score. Scores are
    intentionally not normalized here because dense and sparse scales are part
    of the experiment configuration and must remain auditable.
    """

    def __init__(
        self,
        dense_retriever: SearchRetriever,
        sparse_retriever: SearchRetriever,
        *,
        fusion_strategy: str = FUSION_RRF,
        rrf_k: float = 60.0,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        reranker: Reranker | None = None,
    ):
        if fusion_strategy not in SUPPORTED_FUSION_STRATEGIES:
            allowed = ", ".join(SUPPORTED_FUSION_STRATEGIES)
            raise ValueError(f"fusion_strategy must be one of: {allowed}.")
        if rrf_k < 0:
            raise ValueError("rrf_k must be greater than or equal to 0.")

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_strategy = fusion_strategy
        self.rrf_k = float(rrf_k)
        self.dense_weight = float(dense_weight)
        self.sparse_weight = float(sparse_weight)
        self.reranker = reranker

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if top_k is not None and int(top_k) <= 0:
            return []

        retriever_top_k = None if self.reranker is not None else top_k
        dense_results = self.dense_retriever.search(query, top_k=retriever_top_k)
        sparse_results = self.sparse_retriever.search(query, top_k=retriever_top_k)
        candidates = self._collect_candidates(dense_results, sparse_results)
        fused_results = [self._fused_result(candidate) for candidate in candidates.values()]
        fused_results.sort(key=lambda result: (-result["score"], result["_order"], result["chunk_id"]))

        if self.reranker is None and top_k is not None:
            fused_results = fused_results[: int(top_k)]

        for result in fused_results:
            del result["_order"]

        if self.reranker is not None:
            return self.reranker.rerank(query, fused_results, top_k=top_k)
        return fused_results

    def _collect_candidates(
        self,
        dense_results: list[dict[str, Any]],
        sparse_results: list[dict[str, Any]],
    ) -> dict[str, _Candidate]:
        candidates: dict[str, _Candidate] = {}
        next_order = 0

        for rank, result in enumerate(dense_results or [], start=1):
            chunk_id = _result_chunk_id(result)
            if chunk_id not in candidates:
                candidates[chunk_id] = _Candidate(chunk_id=chunk_id, first_seen_order=next_order)
                next_order += 1
            candidate = candidates[chunk_id]
            candidate.dense_result = result
            candidate.dense_score = _result_score(result)
            candidate.dense_rank = rank

        for rank, result in enumerate(sparse_results or [], start=1):
            chunk_id = _result_chunk_id(result)
            if chunk_id not in candidates:
                candidates[chunk_id] = _Candidate(chunk_id=chunk_id, first_seen_order=next_order)
                next_order += 1
            candidate = candidates[chunk_id]
            candidate.sparse_result = result
            candidate.sparse_score = _result_score(result)
            candidate.sparse_rank = rank

        return candidates

    def _fused_result(self, candidate: _Candidate) -> dict[str, Any]:
        rrf_score = self._rrf_score(candidate)
        weighted_score = self._weighted_score(candidate)
        score = rrf_score if self.fusion_strategy == FUSION_RRF else weighted_score
        representative = _representative_result(candidate)
        metadata = _merged_metadata(candidate, representative)
        source_chunk_id = metadata.get("source_chunk_id") or _get_value(
            representative, "source_chunk_id"
        )

        return {
            "chunk_id": candidate.chunk_id,
            "source_chunk_id": source_chunk_id,
            "score": score,
            "chunk": _result_chunk(representative, candidate.chunk_id),
            "metadata": metadata,
            "diagnostics": {
                "dense_score": candidate.dense_score,
                "sparse_score": candidate.sparse_score,
                "rrf_score": rrf_score,
                "weighted_score": weighted_score,
                "fusion_strategy": self.fusion_strategy,
            },
            "_order": candidate.first_seen_order,
        }

    def _rrf_score(self, candidate: _Candidate) -> float:
        score = 0.0
        if candidate.dense_rank is not None:
            score += 1.0 / (self.rrf_k + candidate.dense_rank)
        if candidate.sparse_rank is not None:
            score += 1.0 / (self.rrf_k + candidate.sparse_rank)
        return score

    def _weighted_score(self, candidate: _Candidate) -> float:
        dense_score = candidate.dense_score if candidate.dense_score is not None else 0.0
        sparse_score = candidate.sparse_score if candidate.sparse_score is not None else 0.0
        return self.dense_weight * dense_score + self.sparse_weight * sparse_score


def _representative_result(candidate: _Candidate) -> dict[str, Any]:
    if candidate.dense_result is None and candidate.sparse_result is None:
        raise ValueError("Candidate has no retriever result.")
    if candidate.dense_result is None:
        return dict(candidate.sparse_result or {})
    if candidate.sparse_result is None:
        return dict(candidate.dense_result)

    dense_rank = candidate.dense_rank or 0
    sparse_rank = candidate.sparse_rank or 0
    if dense_rank <= sparse_rank:
        return dict(candidate.dense_result)
    return dict(candidate.sparse_result)


def _merged_metadata(candidate: _Candidate, representative: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(_get_value(representative, "metadata") or {})
    for result in (candidate.dense_result, candidate.sparse_result):
        if result is None:
            continue
        result_metadata = dict(_get_value(result, "metadata") or {})
        for key, value in result_metadata.items():
            metadata.setdefault(key, value)

    for field in AUDIT_LIST_FIELDS:
        metadata[field] = _merge_lists(
            metadata.get(field),
            _metadata_value(candidate.dense_result, field),
            _metadata_value(candidate.sparse_result, field),
        )

    for field in AUDIT_SCALAR_FIELDS:
        if metadata.get(field) is None:
            fallback = _metadata_value(candidate.dense_result, field)
            if fallback is None:
                fallback = _metadata_value(candidate.sparse_result, field)
            if fallback is not None:
                metadata[field] = fallback

    return metadata


def _metadata_value(result: dict[str, Any] | None, field: str) -> Any:
    if result is None:
        return None
    metadata = _get_value(result, "metadata") or {}
    if isinstance(metadata, dict) and field in metadata:
        return metadata[field]
    return _get_value(result, field)


def _merge_lists(*values: Any) -> list[Any]:
    merged = []
    for value in values:
        if value is None:
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            if item not in merged:
                merged.append(item)
    return merged


def _result_chunk(result: dict[str, Any], chunk_id: str) -> dict[str, Any]:
    chunk = dict(_get_value(result, "chunk") or {})
    chunk.setdefault("chunk_id", chunk_id)
    content = _get_value(result, "content")
    if content is not None:
        chunk.setdefault("content", content)
    return chunk


def _result_chunk_id(result: dict[str, Any]) -> str:
    chunk_id = _get_value(result, "chunk_id")
    if chunk_id is None:
        chunk_id = _get_value(result, "chunk", "chunk_id")
    if chunk_id is None:
        raise ValueError("Retriever result is missing chunk_id.")
    return str(chunk_id)


def _result_score(result: dict[str, Any]) -> float:
    return float(_get_value(result, "score") or 0.0)


def _get_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return None
    return current
