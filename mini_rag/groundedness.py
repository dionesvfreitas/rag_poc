import re
import unicodedata
from typing import Any

from mini_rag.answering import INSUFFICIENT_EVIDENCE_MESSAGE


GROUNDING_CHECKS = (
    "has_answer",
    "citations_exist",
    "citations_in_used_context",
    "used_context_consistent",
    "insufficient_evidence_consistent",
    "prompt_contains_grounding",
    "prompt_contains_context",
    "prompt_contains_chunk_ids",
    "prompt_contains_pages",
    "prompt_contains_sections",
)


class GroundednessValidator:
    """Deterministic validation for RAG answer provenance and prompt integrity."""

    def __init__(self, *, refusal_message: str = INSUFFICIENT_EVIDENCE_MESSAGE):
        self.refusal_message = refusal_message

    def validate(self, answer_result: dict[str, Any]) -> dict[str, Any]:
        result = dict(answer_result or {})
        answer = str(result.get("answer") or "")
        citations = _as_list(result.get("citations"))
        used_context = _as_list(result.get("used_context"))
        insufficient_evidence = bool(result.get("insufficient_evidence"))

        warnings: list[str] = []
        errors: list[str] = []
        unknown_citations: list[str] = []
        citations_outside_context: list[str] = []
        missing_citations: list[str] = []
        checks = {name: True for name in GROUNDING_CHECKS}

        context_ids = _ids_from_items(used_context)
        citation_ids, missing_citation_fields = _ids_from_citations(citations, warnings)
        valid_reference_ids = context_ids | citation_ids

        checks["has_answer"] = bool(answer.strip())
        if not checks["has_answer"]:
            errors.append("answer_empty")

        if missing_citation_fields:
            checks["citations_exist"] = False
            missing_citations.extend(missing_citation_fields)
            errors.append("citations_missing_chunk_id")

        citations_without_context = sorted(citation_ids - context_ids)
        if citations_without_context:
            checks["citations_in_used_context"] = False
            citations_outside_context.extend(citations_without_context)
            errors.append("citations_outside_used_context")

        context_missing_ids = _items_missing_ids(used_context)
        if context_missing_ids:
            checks["used_context_consistent"] = False
            errors.append("used_context_missing_chunk_id")
        if not insufficient_evidence and not context_ids:
            checks["used_context_consistent"] = False
            errors.append("used_context_empty_without_insufficient_evidence")

        if not insufficient_evidence and context_ids and not citations:
            checks["citations_exist"] = False
            missing_citations.append("citations")
            errors.append("citations_required_when_context_is_used")

        unknown_citations.extend(_unknown_chunk_references(answer, valid_reference_ids))
        if unknown_citations:
            errors.append("answer_mentions_unknown_chunk")

        checks["insufficient_evidence_consistent"] = _validate_insufficient_evidence(
            insufficient_evidence=insufficient_evidence,
            answer=answer,
            citation_ids=citation_ids,
            errors=errors,
            refusal_message=self.refusal_message,
        )

        prompt = _prompt_from_result(result)
        if prompt is None:
            warnings.append("prompt_not_available_for_validation")
            for check_name in _prompt_check_names():
                checks[check_name] = False
        else:
            _validate_prompt(prompt, checks, errors)

        return {
            "valid": not errors,
            "checks": checks,
            "unknown_citations": sorted(set(unknown_citations)),
            "citations_outside_context": citations_outside_context,
            "missing_citations": missing_citations,
            "warnings": warnings,
            "errors": errors,
        }


def validate_groundedness(answer_result: dict[str, Any]) -> dict[str, Any]:
    return GroundednessValidator().validate(answer_result)


def _validate_insufficient_evidence(
    *,
    insufficient_evidence: bool,
    answer: str,
    citation_ids: set[str],
    errors: list[str],
    refusal_message: str,
) -> bool:
    if insufficient_evidence:
        consistent = True
        if citation_ids:
            errors.append("insufficient_evidence_has_citations")
            consistent = False
        if not _contains_refusal_message(answer, refusal_message):
            errors.append("insufficient_evidence_missing_refusal_message")
            consistent = False
        return consistent

    if _contains_refusal_message(answer, refusal_message):
        errors.append("answer_refuses_without_insufficient_evidence_flag")
        return False
    return True


def _validate_prompt(prompt: str, checks: dict[str, bool], errors: list[str]) -> None:
    normalized = _normalize_text(prompt)
    checks["prompt_contains_grounding"] = _prompt_contains_grounding(normalized)
    checks["prompt_contains_context"] = "contexto" in normalized and "pergunta" in normalized
    checks["prompt_contains_chunk_ids"] = bool(
        re.search(r"\bchunk_id\s*:\s*\S+", prompt, flags=re.IGNORECASE)
    )
    checks["prompt_contains_pages"] = bool(
        re.search(r"\bpages?\s*:\s*\d+", prompt, flags=re.IGNORECASE)
        or re.search(r"\bpage_(?:start|end)\b", prompt, flags=re.IGNORECASE)
        or "pagina" in normalized
    )
    checks["prompt_contains_sections"] = bool(
        re.search(r"\bsection_path\s*:", prompt, flags=re.IGNORECASE)
        or "secao" in normalized
        or "secoes" in normalized
    )

    for check_name in _prompt_check_names():
        if not checks[check_name]:
            errors.append(f"{check_name}_missing")


def _prompt_contains_grounding(normalized_prompt: str) -> bool:
    grounding_signals = (
        "somente usando",
        "apenas usando",
        "nao use conhecimento externo",
        "cite apenas fontes",
        "fontes fornecidas",
        "contexto recuperado",
    )
    refusal_signals = (
        "evidencia suficiente",
        "nao encontrei evidencia suficiente",
        "recuse",
        "nao responda",
    )
    return any(signal in normalized_prompt for signal in grounding_signals) and any(
        signal in normalized_prompt for signal in refusal_signals
    )


def _ids_from_items(items: list[Any]) -> set[str]:
    return {item_id for item_id in (_item_chunk_id(item) for item in items) if item_id}


def _ids_from_citations(citations: list[Any], warnings: list[str]) -> tuple[set[str], list[str]]:
    citation_ids: set[str] = set()
    missing_fields: list[str] = []

    for index, citation in enumerate(citations):
        chunk_id = _item_chunk_id(citation)
        if not chunk_id:
            missing_fields.append(f"citations[{index}].chunk_id")
            continue

        citation_ids.add(chunk_id)
        if _get_value(citation, "source_chunk_id") in (None, ""):
            warnings.append(f"citation_missing_source_chunk_id:{chunk_id}")
        if (
            _get_value(citation, "page_start") in (None, "")
            and _get_value(citation, "page_end") in (None, "")
            and not _as_list(_get_value(citation, "pages"))
        ):
            warnings.append(f"citation_missing_pages:{chunk_id}")
        if _get_value(citation, "section_path") in (None, ""):
            warnings.append(f"citation_missing_section_path:{chunk_id}")

    return citation_ids, missing_fields


def _items_missing_ids(items: list[Any]) -> list[str]:
    return [f"used_context[{index}].chunk_id" for index, item in enumerate(items) if not _item_chunk_id(item)]


def _item_chunk_id(item: Any) -> str | None:
    chunk_id = _first_present_value(
        item,
        (
            ("chunk_id",),
            ("chunk", "chunk_id"),
            ("metadata", "chunk_id"),
            ("id",),
        ),
    )
    if chunk_id in (None, ""):
        return None
    return str(chunk_id)


def _unknown_chunk_references(answer: str, valid_reference_ids: set[str]) -> list[str]:
    references = set()
    for pattern in (
        r"\[((?:chunk[_-]?[A-Za-z0-9_-]+)|(?:c\d+[A-Za-z0-9_-]*))\]",
        r"(?<![\w-])((?:chunk[_-]?[A-Za-z0-9_-]+)|(?:c\d+[A-Za-z0-9_-]*))(?![\w-])",
    ):
        for match in re.finditer(pattern, answer, flags=re.IGNORECASE):
            references.add(match.group(1))
    return sorted(reference for reference in references if reference not in valid_reference_ids)


def _prompt_from_result(result: dict[str, Any]) -> str | None:
    metadata = result.get("metadata") or {}
    for key in ("prompt", "answer_prompt", "generation_prompt", "last_prompt"):
        prompt = _get_value(metadata, key)
        if isinstance(prompt, str) and prompt.strip():
            return prompt
    for nested_key in ("answering", "generation", "llm"):
        nested = _get_value(metadata, nested_key)
        if isinstance(nested, dict):
            prompt = _prompt_from_result({"metadata": nested})
            if prompt is not None:
                return prompt
    return None


def _contains_refusal_message(answer: str, refusal_message: str) -> bool:
    normalized_answer = _normalize_text(answer)
    normalized_refusal = _normalize_text(refusal_message)
    return normalized_refusal in normalized_answer


def _prompt_check_names() -> tuple[str, ...]:
    return (
        "prompt_contains_grounding",
        "prompt_contains_context",
        "prompt_contains_chunk_ids",
        "prompt_contains_pages",
        "prompt_contains_sections",
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


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


def _normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return ascii_text.lower()
