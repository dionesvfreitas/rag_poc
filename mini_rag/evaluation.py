import json
from pathlib import Path
from typing import Any


TOP_K_VALUES = (1, 3, 5)
MIN_EVALUATION_TOP_K = max(TOP_K_VALUES)
MATCH_SOURCE_CHUNK_ID = "source_chunk_id"
MATCH_CHUNK_ID = "chunk_id"
MATCH_NO_MATCH = "no_match"


def load_evaluation_dataset(path) -> dict[str, Any]:
    dataset_path = Path(path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    validate_evaluation_dataset(dataset)
    return dataset


def validate_evaluation_dataset(dataset: dict[str, Any]) -> None:
    required_fields = ("schema_version", "name", "cases")
    for field in required_fields:
        if field not in dataset:
            raise ValueError(f"Evaluation dataset is missing required field: {field}")

    cases = dataset["cases"]
    if not isinstance(cases, list):
        raise ValueError("Evaluation dataset field 'cases' must be a list.")

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Evaluation case #{index + 1} must be an object.")
        case_label = case.get("id") or f"#{index + 1}"
        for field in ("id", "question", "expected"):
            if field not in case:
                raise ValueError(f"Evaluation case {case_label} is missing required field: {field}")

        expected = case["expected"]
        if not isinstance(expected, dict):
            raise ValueError(f"Evaluation case {case_label} field 'expected' must be an object.")

        if not _expected_ids(expected, MATCH_SOURCE_CHUNK_ID) and not _expected_ids(expected, MATCH_CHUNK_ID):
            raise ValueError(
                f"Evaluation case {case_label} must define expected.source_chunk_ids "
                "or expected.chunk_ids."
            )


def validate_top_k(top_k: int) -> int:
    evaluated_top_k = int(top_k)
    if evaluated_top_k < MIN_EVALUATION_TOP_K:
        raise ValueError(
            f"top_k must be >= {MIN_EVALUATION_TOP_K}; got {evaluated_top_k}. "
            "Values below 5 make top5_hit and recall@5 ambiguous."
        )
    return evaluated_top_k


def expected_match_basis(expected: dict[str, Any]) -> str:
    if _expected_ids(expected, MATCH_SOURCE_CHUNK_ID):
        return MATCH_SOURCE_CHUNK_ID
    if _expected_ids(expected, MATCH_CHUNK_ID):
        return MATCH_CHUNK_ID
    raise ValueError("Expected relevance must define source_chunk_ids or chunk_ids.")


def expected_item_ids(expected: dict[str, Any]) -> list[str]:
    return _unique_preserving_order(_expected_ids(expected, expected_match_basis(expected)))


def match_result(result: Any, expected: dict[str, Any]) -> dict[str, Any]:
    basis = expected_match_basis(expected)
    expected_ids = set(expected_item_ids(expected))
    source_chunk_id = _result_source_chunk_id(result)
    chunk_id = _result_chunk_id(result)

    if basis == MATCH_SOURCE_CHUNK_ID:
        matched = source_chunk_id in expected_ids
        match_type = MATCH_SOURCE_CHUNK_ID if matched else MATCH_NO_MATCH
    else:
        matched = chunk_id in expected_ids
        match_type = MATCH_CHUNK_ID if matched else MATCH_NO_MATCH

    return {
        "matched": matched,
        "match_type": match_type,
        "chunk_id": chunk_id,
        "source_chunk_id": source_chunk_id,
    }


def evaluate_retrieval(
    dataset: dict[str, Any],
    retriever: Any,
    *,
    top_k: int | None = None,
    dataset_path: str | Path | None = None,
    index_path: str | Path | None = None,
    index_metadata: dict[str, Any] | None = None,
    similarity_threshold: float | None = None,
) -> dict[str, Any]:
    validate_evaluation_dataset(dataset)
    default_top_k = dataset.get("default_top_k", MIN_EVALUATION_TOP_K)
    evaluation_top_k = validate_top_k(top_k if top_k is not None else default_top_k)

    case_results = [
        evaluate_retrieval_case(
            case,
            retriever.search(case["question"], top_k=evaluation_top_k),
            top_k=evaluation_top_k,
        )
        for case in dataset["cases"]
    ]
    metrics = summarize_metrics(case_results)
    metrics["hits_by_section_path"] = summarize_hits_by_section_path(case_results)
    metrics["hits_by_page"] = summarize_hits_by_page(case_results)

    return {
        "schema_version": "1.0",
        "dataset": _dataset_metadata(dataset, dataset_path),
        "index": _index_metadata(retriever, index_path, index_metadata),
        "config": {
            "top_k": evaluation_top_k,
            "similarity_threshold": _config_similarity_threshold(retriever, similarity_threshold),
        },
        "metrics": metrics,
        "retrieval_diagnostics": summarize_retrieval_diagnostics(case_results),
        "cases": case_results,
    }


def evaluate_retrieval_case(
    case: dict[str, Any],
    results: list[Any],
    *,
    top_k: int = max(TOP_K_VALUES),
) -> dict[str, Any]:
    top_k = validate_top_k(top_k)
    expected = case["expected"]
    basis = expected_match_basis(expected)
    expected_ids = expected_item_ids(expected)
    expected_id_set = set(expected_ids)
    evaluated_results = []
    first_relevant_rank = None
    first_hit_match_type = MATCH_NO_MATCH

    retrieved_ids_by_k = {k: set() for k in TOP_K_VALUES}

    for rank, result in enumerate(list(results)[:top_k], start=1):
        match = match_result(result, expected)
        matched_id = match["source_chunk_id"] if basis == MATCH_SOURCE_CHUNK_ID else match["chunk_id"]

        if match["matched"] and first_relevant_rank is None:
            first_relevant_rank = rank
            first_hit_match_type = match["match_type"]

        if match["matched"]:
            for k in TOP_K_VALUES:
                if rank <= k:
                    retrieved_ids_by_k[k].add(matched_id)

        evaluated_results.append(_report_result(rank, result, match))

    expected_count = len(expected_id_set)
    case_result = {
        "id": case["id"],
        "question": case["question"],
        "expected": expected,
        "expected_match_basis": basis,
        "expected_count": expected_count,
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": (1.0 / first_relevant_rank) if first_relevant_rank else 0.0,
        "first_hit_match_type": first_hit_match_type,
        "results": evaluated_results,
    }

    for k in TOP_K_VALUES:
        retrieved_count = len(retrieved_ids_by_k[k] & expected_id_set)
        case_result[f"top{k}_hit"] = bool(first_relevant_rank and first_relevant_rank <= k)
        case_result[f"retrieved_expected_count@{k}"] = retrieved_count
        case_result[f"recall@{k}"] = retrieved_count / expected_count if expected_count else 0.0

    return case_result


def summarize_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not case_results:
        return {
            "top1_hit": 0.0,
            "top3_hit": 0.0,
            "top5_hit": 0.0,
            "mrr": 0.0,
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "hits_by_section_path": {},
            "hits_by_page": {},
        }

    return {
        "top1_hit": _average(case_result["top1_hit"] for case_result in case_results),
        "top3_hit": _average(case_result["top3_hit"] for case_result in case_results),
        "top5_hit": _average(case_result["top5_hit"] for case_result in case_results),
        "mrr": _average(case_result["reciprocal_rank"] for case_result in case_results),
        "recall@1": _average(case_result["recall@1"] for case_result in case_results),
        "recall@3": _average(case_result["recall@3"] for case_result in case_results),
        "recall@5": _average(case_result["recall@5"] for case_result in case_results),
    }


def summarize_retrieval_diagnostics(case_results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total_questions": len(case_results),
        "questions_with_no_hit": sum(1 for case_result in case_results if case_result["first_relevant_rank"] is None),
        "questions_hit_via_source_chunk_id": sum(
            1 for case_result in case_results if case_result["first_hit_match_type"] == MATCH_SOURCE_CHUNK_ID
        ),
        "questions_hit_via_chunk_id": sum(
            1 for case_result in case_results if case_result["first_hit_match_type"] == MATCH_CHUNK_ID
        ),
        "questions_hit_only_via_chunk_id": sum(
            1 for case_result in case_results if case_result["expected_match_basis"] == MATCH_CHUNK_ID
        ),
        "questions_with_multiple_expected_items": sum(
            1 for case_result in case_results if case_result["expected_count"] > 1
        ),
    }


def summarize_hits_by_section_path(case_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for case_result in case_results:
        if case_result["first_relevant_rank"] is None:
            continue
        for section_path in _expected_section_keys(case_result["expected"]):
            buckets.setdefault(section_path, []).append(case_result)
    return _summarize_diagnostic_buckets(buckets)


def summarize_hits_by_page(case_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for case_result in case_results:
        if case_result["first_relevant_rank"] is None:
            continue
        for page in _expected_page_keys(case_result["expected"]):
            buckets.setdefault(page, []).append(case_result)
    return _summarize_diagnostic_buckets(buckets)


def _summarize_diagnostic_buckets(
    buckets: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    diagnostics = {}
    for key in sorted(buckets):
        bucket_cases = buckets[key]
        diagnostics[key] = {
            "expected_cases": len(bucket_cases),
            "top1_hits": sum(1 for case_result in bucket_cases if case_result["top1_hit"]),
            "top3_hits": sum(1 for case_result in bucket_cases if case_result["top3_hit"]),
            "top5_hits": sum(1 for case_result in bucket_cases if case_result["top5_hit"]),
            "recall@1": _average(case_result["recall@1"] for case_result in bucket_cases),
            "recall@3": _average(case_result["recall@3"] for case_result in bucket_cases),
            "recall@5": _average(case_result["recall@5"] for case_result in bucket_cases),
        }
    return diagnostics


def _average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _expected_ids(expected: dict[str, Any], basis: str) -> list[str]:
    key = "source_chunk_ids" if basis == MATCH_SOURCE_CHUNK_ID else "chunk_ids"
    values = expected.get(key) or []
    if not isinstance(values, list):
        raise ValueError(f"expected.{key} must be a list when provided.")
    return [value for value in values if value]


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _result_source_chunk_id(result: Any) -> str | None:
    return _get_value(result, "metadata", "source_chunk_id")


def _result_chunk_id(result: Any) -> str | None:
    return _get_value(result, "chunk", "chunk_id")


def _report_result(rank: int, result: Any, match: dict[str, Any]) -> dict[str, Any]:
    report_result = {
        "rank": rank,
        "score": _get_value(result, "score"),
        "chunk_id": match["chunk_id"],
        "source_chunk_id": match["source_chunk_id"],
        "section_path": _result_section_path(result),
        "pages": _result_pages(result),
        "matched": match["matched"],
        "match_type": match["match_type"],
    }
    content_preview = _content_preview(result)
    if content_preview is not None:
        report_result["content_preview"] = content_preview[:200]
    return report_result


def _dataset_metadata(dataset: dict[str, Any], dataset_path: str | Path | None) -> dict[str, Any]:
    return {
        "name": dataset.get("name"),
        "path": str(dataset_path) if dataset_path is not None else dataset.get("path"),
        "case_count": len(dataset.get("cases") or []),
    }


def _index_metadata(
    retriever: Any,
    index_path: str | Path | None,
    index_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    vector_store = getattr(retriever, "vector_store", None)
    metadata = {
        "path": str(index_path) if index_path is not None else None,
        "embedding_model": getattr(vector_store, "embedding_model", None),
        "source_path": getattr(vector_store, "source_path", None),
    }
    metadata.update(index_metadata or {})
    if index_path is not None:
        metadata["path"] = str(index_path)
    return metadata


def _config_similarity_threshold(retriever: Any, similarity_threshold: float | None) -> float | None:
    if similarity_threshold is not None:
        return similarity_threshold
    return getattr(retriever, "similarity_threshold", None)


def _result_section_path(result: Any) -> list[str]:
    section_path = _get_value(result, "metadata", "section_path")
    if section_path is None:
        return []
    if isinstance(section_path, list):
        return [str(part) for part in section_path]
    return [str(section_path)]


def _result_pages(result: Any) -> list[int | str]:
    pages = _get_value(result, "metadata", "pages")
    if isinstance(pages, list):
        return pages
    if pages is not None:
        return [pages]

    page_start = _get_value(result, "metadata", "page_start")
    page_end = _get_value(result, "metadata", "page_end")
    if page_start is None:
        return []
    if page_end is None:
        return [page_start]
    if isinstance(page_start, int) and isinstance(page_end, int) and page_end >= page_start:
        return list(range(page_start, page_end + 1))
    return [page_start]


def _content_preview(result: Any) -> str | None:
    preview = _get_value(result, "content_preview")
    if preview is None:
        preview = _get_value(result, "chunk", "content_preview")
    if preview is None:
        preview = _get_value(result, "metadata", "content_preview")
    if preview is None:
        return None
    return str(preview)


def _expected_section_keys(expected: dict[str, Any]) -> list[str]:
    section_paths = expected.get("section_paths") or []
    keys = []
    for section_path in section_paths:
        if isinstance(section_path, list):
            key = " > ".join(str(part) for part in section_path if part)
        else:
            key = str(section_path)
        if key:
            keys.append(key)
    return _unique_preserving_order(keys)


def _expected_page_keys(expected: dict[str, Any]) -> list[str]:
    return _unique_preserving_order([str(page) for page in expected.get("pages") or []])


def _get_value(value: Any, *path: str) -> Any:
    current = value
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current
