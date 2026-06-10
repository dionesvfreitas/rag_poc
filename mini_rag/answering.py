import json
import re
from typing import Any, Protocol
from urllib import request
from urllib.error import URLError

from mini_rag.citations import CitationBuilder


INSUFFICIENT_EVIDENCE_MESSAGE = "Não encontrei evidência suficiente nos trechos recuperados."
DEFAULT_CONTEXT_CHAR_LIMIT = 2400


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class FakeLLMProvider:
    """Deterministic LLM provider for tests and local experiments."""

    model = "fake"

    def __init__(self, response: str | None = None, *, insufficient: bool = False):
        self.response = response
        self.insufficient = insufficient
        self.prompts: list[str] = []
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.call_count += 1

        if self.response is not None:
            return self.response
        if self.insufficient:
            return INSUFFICIENT_EVIDENCE_MESSAGE

        chunk_id = _first_chunk_id_from_prompt(prompt)
        if chunk_id is None:
            return INSUFFICIENT_EVIDENCE_MESSAGE
        return f"Resposta baseada nos trechos recuperados. [{chunk_id}]"


class OllamaLLMProvider:
    """Optional Ollama provider using a local HTTP endpoint.

    This provider is intentionally isolated from tests. It uses only the
    standard library and raises clear runtime errors when Ollama is not
    available or returns an unexpected response.
    """

    def __init__(
        self,
        model: str = "llama3.1",
        *,
        endpoint: str = "http://localhost:11434/api/generate",
        timeout: float = 30.0,
    ):
        self.model = model
        self.endpoint = endpoint
        self.timeout = float(timeout)

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")
        http_request = request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(
                f"Ollama is unavailable at {self.endpoint}. Start Ollama or "
                "inject a different LLMProvider."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc

        text = data.get("response")
        if not isinstance(text, str):
            raise RuntimeError("Ollama response does not contain a text response.")
        return text


class AnswerGenerator:
    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        *,
        evidence_score_threshold: float | None = None,
        citation_builder: CitationBuilder | None = None,
        refusal_message: str = INSUFFICIENT_EVIDENCE_MESSAGE,
        context_char_limit: int = DEFAULT_CONTEXT_CHAR_LIMIT,
    ):
        self.llm_provider = llm_provider or FakeLLMProvider()
        self.evidence_score_threshold = evidence_score_threshold
        self.citation_builder = citation_builder or CitationBuilder()
        self.refusal_message = refusal_message
        self.context_char_limit = int(context_char_limit)
        self.last_prompt: str | None = None

    def answer(
        self,
        question: str,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        search_results = list(results or [])
        if not search_results:
            return self._insufficient_response()
        if self._all_scores_below_threshold(search_results):
            return self._insufficient_response()

        used_context = [
            _context_from_result(result, rank=rank, content_limit=self.context_char_limit)
            for rank, result in enumerate(search_results, start=1)
        ]
        prompt = build_answer_prompt(question, used_context, refusal_message=self.refusal_message)
        self.last_prompt = prompt
        answer_text = self.llm_provider.generate(prompt)
        valid_chunk_ids = {context["chunk_id"] for context in used_context}
        answer_text = _remove_unknown_chunk_references(answer_text, valid_chunk_ids)

        return {
            "answer": answer_text,
            "citations": self.citation_builder.build(search_results),
            "used_context": used_context,
            "insufficient_evidence": answer_text.strip() == self.refusal_message,
            "metadata": {
                "model": _provider_model_name(self.llm_provider),
                "context_count": len(used_context),
            },
        }

    def _insufficient_response(self) -> dict[str, Any]:
        return {
            "answer": self.refusal_message,
            "citations": [],
            "used_context": [],
            "insufficient_evidence": True,
            "metadata": {
                "model": _provider_model_name(self.llm_provider),
                "context_count": 0,
            },
        }

    def _all_scores_below_threshold(self, results: list[dict[str, Any]]) -> bool:
        if self.evidence_score_threshold is None:
            return False
        threshold = float(self.evidence_score_threshold)
        return all(_result_score(result) < threshold for result in results)


def build_answer_prompt(
    question: str,
    used_context: list[dict[str, Any]],
    *,
    refusal_message: str = INSUFFICIENT_EVIDENCE_MESSAGE,
) -> str:
    context_blocks = [_format_context_block(context) for context in used_context]
    context_text = "\n\n".join(context_blocks)
    return (
        "Você é um gerador de respostas RAG auditável.\n"
        "Responda somente usando os trechos fornecidos.\n"
        "Não use conhecimento externo.\n"
        f"Se os trechos não contiverem evidência suficiente, responda: \"{refusal_message}\"\n"
        "Sempre cite os chunk_ids usados.\n"
        "Cite apenas fontes fornecidas no contexto recuperado.\n\n"
        f"Pergunta do usuário:\n{question}\n\n"
        f"Contexto recuperado:\n{context_text}\n\n"
        "Resposta final:"
    )


def _format_context_block(context: dict[str, Any]) -> str:
    lines = [
        f"[rank {context['rank']}]",
        f"score: {context['score']}",
        f"chunk_id: {context['chunk_id']}",
        f"source_chunk_id: {context['source_chunk_id']}",
        f"pages: {context['page_start']}-{context['page_end']}",
        f"section_path: {_compact_json(context['section_path'])}",
    ]
    if context.get("source_block_ids"):
        lines.append(f"source_block_ids: {_compact_json(context['source_block_ids'])}")
    if context.get("source_spans"):
        lines.append(f"source_spans_count: {len(context['source_spans'])}")
    if context.get("related_assets"):
        lines.append(f"related_assets: {_compact_json(context['related_assets'])}")
    lines.append(f"content:\n{context['content']}")
    return "\n".join(lines)


def _context_from_result(
    result: dict[str, Any],
    *,
    rank: int,
    content_limit: int,
) -> dict[str, Any]:
    metadata = dict(_get_value(result, "metadata") or {})
    return {
        "rank": rank,
        "score": _get_value(result, "score"),
        "chunk_id": _result_chunk_id(result),
        "source_chunk_id": _first_present_value(
            result,
            (
                ("source_chunk_id",),
                ("metadata", "source_chunk_id"),
            ),
        ),
        "page_start": _first_present_value(
            result,
            (
                ("page_start",),
                ("metadata", "page_start"),
                ("metadata", "page_no"),
            ),
        ),
        "page_end": _first_present_value(
            result,
            (
                ("page_end",),
                ("metadata", "page_end"),
                ("metadata", "page_no"),
            ),
        ),
        "section_path": list(_metadata_or_result_value(result, metadata, "section_path") or []),
        "content": _truncate(_result_content(result), content_limit),
        "source_block_ids": list(
            _metadata_or_result_value(result, metadata, "source_block_ids") or []
        ),
        "source_spans": list(_metadata_or_result_value(result, metadata, "source_spans") or []),
        "related_assets": list(
            _metadata_or_result_value(result, metadata, "related_assets") or []
        ),
    }


def _result_chunk_id(result: dict[str, Any]) -> str:
    chunk_id = _first_present_value(
        result,
        (
            ("chunk_id",),
            ("chunk", "chunk_id"),
        ),
    )
    if chunk_id is None:
        raise ValueError("Answer context result is missing chunk_id.")
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


def _result_score(result: dict[str, Any]) -> float:
    score = _get_value(result, "score")
    return float(score) if score is not None else 0.0


def _provider_model_name(provider: LLMProvider) -> str:
    model = getattr(provider, "model", None)
    if model:
        return str(model)
    return provider.__class__.__name__


def _first_present_value(value: Any, paths: tuple[tuple[str, ...], ...]) -> Any:
    for path in paths:
        field_value = _get_value(value, *path)
        if field_value is not None:
            return field_value
    return None


def _metadata_or_result_value(
    result: dict[str, Any],
    metadata: dict[str, Any],
    field: str,
) -> Any:
    if field in metadata:
        return metadata.get(field)
    return result.get(field)


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


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[trecho truncado]"


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _first_chunk_id_from_prompt(prompt: str) -> str | None:
    match = re.search(r"^chunk_id:\s*(\S+)", prompt, flags=re.MULTILINE)
    return match.group(1) if match else None


def _remove_unknown_chunk_references(text: str, valid_chunk_ids: set[str]) -> str:
    if not valid_chunk_ids:
        return text

    def replace_square(match: re.Match[str]) -> str:
        reference = match.group(1).strip()
        normalized = reference
        if normalized.lower().startswith("chunk_id:"):
            normalized = normalized.split(":", 1)[1].strip()
        if normalized.lower().startswith("chunk:"):
            normalized = normalized.split(":", 1)[1].strip()
        return match.group(0) if normalized in valid_chunk_ids else ""

    sanitized = re.sub(r"\[([^\[\]]+)\]", replace_square, text)
    sanitized = re.sub(
        r"\(chunk_id:\s*([^)]+)\)",
        lambda match: match.group(0) if match.group(1).strip() in valid_chunk_ids else "",
        sanitized,
        flags=re.IGNORECASE,
    )
    return re.sub(r" {2,}", " ", sanitized).strip()
