import re
from typing import Any, Protocol


TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        ...


class IdentityReranker:
    """Safe reranker that preserves input order while adding diagnostics."""

    name = "identity"

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        del query
        annotated = [
            _result_with_rerank_diagnostics(
                result,
                rerank_score=_original_score(result),
                reranker=self.name,
            )
            for result in (results or [])
        ]
        return _limit_results(annotated, top_k)


class FakeReranker:
    """Deterministic reranker for tests and local experiments.

    Scores are resolved in this order:
    1. explicit ``scores_by_chunk_id`` mapping;
    2. ``score_key`` found on the result, metadata, or diagnostics;
    3. lexical overlap between query tokens and result content.
    """

    name = "fake"

    def __init__(
        self,
        *,
        scores_by_chunk_id: dict[str, float] | None = None,
        score_key: str = "fake_rerank_score",
        default_score: float = 0.0,
    ):
        self.scores_by_chunk_id = {
            str(chunk_id): float(score)
            for chunk_id, score in (scores_by_chunk_id or {}).items()
        }
        self.score_key = score_key
        self.default_score = float(default_score)

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        scored = []
        for order, result in enumerate(results or []):
            score = self._score(query, result)
            scored.append(
                (
                    score,
                    order,
                    _result_with_rerank_diagnostics(
                        result,
                        rerank_score=score,
                        reranker=self.name,
                    ),
                )
            )

        scored.sort(key=lambda item: (-item[0], item[1]))
        return _limit_results([result for _, _, result in scored], top_k)

    def _score(self, query: str, result: dict[str, Any]) -> float:
        chunk_id = _result_chunk_id(result)
        if chunk_id in self.scores_by_chunk_id:
            return self.scores_by_chunk_id[chunk_id]

        injected_score = _first_present_value(
            result,
            (
                (self.score_key,),
                ("metadata", self.score_key),
                ("diagnostics", self.score_key),
            ),
        )
        if injected_score is not None:
            return float(injected_score)

        query_terms = set(_tokenize(query))
        if not query_terms:
            return self.default_score

        content_terms = _tokenize(_result_content(result))
        if not content_terms:
            return self.default_score

        overlap = sum(1 for term in content_terms if term in query_terms)
        return float(overlap) if overlap else self.default_score


class CrossEncoderReranker:
    """Optional cross-encoder reranker with lazy sentence-transformers import."""

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        model: Any | None = None,
    ):
        self.model_name = model_name
        self._model = model

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if top_k is not None and int(top_k) <= 0:
            return []
        if not results:
            return []

        model = self._load_model()
        pairs = [(query, _result_content(result)) for result in (results or [])]
        raw_scores = model.predict(pairs) if pairs else []
        scored = []

        for order, (result, raw_score) in enumerate(zip(results or [], raw_scores)):
            score = float(raw_score)
            scored.append(
                (
                    score,
                    order,
                    _result_with_rerank_diagnostics(
                        result,
                        rerank_score=score,
                        reranker=self.name,
                    ),
                )
            )

        scored.sort(key=lambda item: (-item[0], item[1]))
        return _limit_results([result for _, _, result in scored], top_k)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "CrossEncoderReranker requires sentence-transformers with "
                "CrossEncoder available. Install the dependency or inject a "
                "compatible model."
            ) from exc

        try:
            self._model = CrossEncoder(self.model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load cross-encoder model '{self.model_name}'. "
                "Use IdentityReranker/FakeReranker for local tests or provide "
                "an injected model."
            ) from exc
        return self._model


def _result_with_rerank_diagnostics(
    result: dict[str, Any],
    *,
    rerank_score: float,
    reranker: str,
) -> dict[str, Any]:
    reranked = dict(result)
    diagnostics = dict(result.get("diagnostics") or {})
    diagnostics.setdefault("original_score", result.get("score"))
    diagnostics["rerank_score"] = float(rerank_score)
    diagnostics["reranker"] = reranker
    reranked["diagnostics"] = diagnostics
    return reranked


def _limit_results(
    results: list[dict[str, Any]],
    top_k: int | None,
) -> list[dict[str, Any]]:
    if top_k is None:
        return results
    limit = int(top_k)
    if limit <= 0:
        return []
    return results[:limit]


def _original_score(result: dict[str, Any]) -> float:
    score = result.get("score")
    return float(score) if score is not None else 0.0


def _result_chunk_id(result: dict[str, Any]) -> str:
    chunk_id = _get_value(result, "chunk_id")
    if chunk_id is None:
        chunk_id = _get_value(result, "chunk", "chunk_id")
    if chunk_id is None:
        raise ValueError("Rerank result is missing chunk_id.")
    return str(chunk_id)


def _result_content(result: dict[str, Any]) -> str:
    content = _first_present_value(
        result,
        (
            ("content",),
            ("text",),
            ("page_content",),
            ("chunk", "content"),
            ("chunk", "text"),
            ("chunk", "page_content"),
        ),
    )
    return str(content or "")


def _first_present_value(value: Any, paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        field_value = _get_value(value, *path)
        if field_value is not None:
            return field_value
    return None


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


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())
