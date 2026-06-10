import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
AUDIT_METADATA_FIELDS = (
    "chunk_id",
    "source_chunk_id",
    "source_spans",
    "source_block_ids",
    "page_start",
    "page_end",
    "section_path",
    "related_assets",
)


def tokenize(text: str) -> list[str]:
    """Tokenize text with deterministic Unicode alphanumeric terms.

    Accents are preserved intentionally: "contratacao" and "contratação"
    are different tokens. This keeps the implementation simple and explicit
    until accent folding or stemming are introduced as separate features.
    """

    return TOKEN_PATTERN.findall((text or "").lower())


@dataclass(frozen=True)
class SparseDocument:
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    terms: Counter[str]
    length: int
    order: int


class SparseRetriever:
    """Local BM25 retriever over existing chunk or index records.

    Results with a zero BM25 score are filtered out. A query with no lexical
    match therefore returns an empty list, which avoids surfacing unrelated
    chunks as false evidence.
    """

    def __init__(self, records: list[Any], *, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0.")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.k1 = float(k1)
        self.b = float(b)
        self.documents = [
            _document_from_record(record, order=index)
            for index, record in enumerate(records or [])
        ]
        self.avgdl = _average(document.length for document in self.documents)
        self.document_frequencies = _document_frequencies(self.documents)
        self.document_count = len(self.documents)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        limit = self.document_count if top_k is None else int(top_k)
        if limit <= 0:
            return []

        query_terms = _unique_preserving_order(tokenize(query))
        if not query_terms:
            return []

        scored = []
        for document in self.documents:
            score = self._score_document(query_terms, document)
            if score <= 0.0:
                continue
            scored.append((score, document.order, document))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [_result(score, document) for score, _, document in scored[:limit]]

    def _score_document(self, query_terms: list[str], document: SparseDocument) -> float:
        if document.length == 0 or self.document_count == 0:
            return 0.0

        score = 0.0
        for term in query_terms:
            tf = document.terms.get(term, 0)
            if tf == 0:
                continue

            idf = self._idf(term)
            denominator = tf + self.k1 * (1 - self.b + self.b * document.length / self.avgdl)
            score += idf * (tf * (self.k1 + 1)) / denominator
        return score

    def _idf(self, term: str) -> float:
        document_frequency = self.document_frequencies.get(term, 0)
        return math.log(1 + (self.document_count - document_frequency + 0.5) / (document_frequency + 0.5))


def _document_from_record(record: Any, *, order: int) -> SparseDocument:
    chunk_id = _record_chunk_id(record)
    content = _record_content(record)
    metadata = _record_metadata(record)
    terms = Counter(tokenize(content))
    return SparseDocument(
        chunk_id=chunk_id,
        content=content,
        metadata=metadata,
        terms=terms,
        length=sum(terms.values()),
        order=order,
    )


def _record_chunk_id(record: Any) -> str:
    chunk_id = _get_value(record, "chunk_id")
    if chunk_id is not None:
        return str(chunk_id)
    chunk_id = _get_value(record, "chunk", "chunk_id")
    if chunk_id is not None:
        return str(chunk_id)
    raise ValueError("Sparse record is missing chunk_id.")


def _record_content(record: Any) -> str:
    direct_content = _first_present_value(record, ("content", "text", "page_content"))
    if direct_content is not None:
        return str(direct_content)

    chunk_content = _first_present_value(
        _get_value(record, "chunk") or {},
        ("content", "text", "page_content"),
    )
    return str(chunk_content or "")


def _record_metadata(record: Any) -> dict[str, Any]:
    metadata = dict(_get_value(record, "metadata") or {})

    for field in AUDIT_METADATA_FIELDS:
        if field == "chunk_id":
            continue
        value = _get_value(record, field)
        if value is not None:
            metadata[field] = value

    metadata.setdefault("source_spans", list(metadata.get("source_spans") or []))
    metadata.setdefault("source_block_ids", list(metadata.get("source_block_ids") or []))
    metadata.setdefault("section_path", list(metadata.get("section_path") or []))
    metadata.setdefault("related_assets", list(metadata.get("related_assets") or []))
    return metadata


def _document_frequencies(documents: list[SparseDocument]) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for document in documents:
        for term in document.terms:
            frequencies[term] = frequencies.get(term, 0) + 1
    return frequencies


def _result(score: float, document: SparseDocument) -> dict[str, Any]:
    return {
        "score": score,
        "chunk": {
            "chunk_id": document.chunk_id,
            "content": document.content,
        },
        "metadata": dict(document.metadata),
    }


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


def _first_present_value(value: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        field_value = _get_value(value, key)
        if field_value is not None:
            return field_value
    return None


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _average(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
