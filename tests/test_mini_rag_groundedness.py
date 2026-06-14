import copy
import unittest

from mini_rag.answering import INSUFFICIENT_EVIDENCE_MESSAGE, build_answer_prompt
from mini_rag.groundedness import GroundednessValidator, validate_groundedness


def used_context(chunk_id="c1", *, source_spans=None, related_assets=None):
    return {
        "rank": 1,
        "score": 0.9,
        "chunk_id": chunk_id,
        "source_chunk_id": f"source-{chunk_id}",
        "page_start": 2,
        "page_end": 3,
        "section_path": ["1 DO OBJETO"],
        "source_block_ids": ["b1"],
        "source_spans": list(source_spans or []),
        "related_assets": list(related_assets or []),
        "content": f"Conteúdo auditável do {chunk_id}.",
    }


def citation(chunk_id="c1", *, source_spans=None, related_assets=None):
    return {
        "chunk_id": chunk_id,
        "source_chunk_id": f"source-{chunk_id}",
        "pages": [2, 3],
        "page_start": 2,
        "page_end": 3,
        "section_path": ["1 DO OBJETO"],
        "source_block_ids": ["b1"],
        "source_spans": list(source_spans or []),
        "related_assets": list(related_assets or []),
        "score": 0.9,
    }


def valid_prompt(contexts=None):
    return build_answer_prompt("Qual é o objeto?", list(contexts or [used_context("c1")]))


def answer_result(
    *,
    answer="Resposta baseada no contexto. [c1]",
    citations=None,
    contexts=None,
    insufficient_evidence=False,
    prompt=True,
):
    contexts = list(contexts if contexts is not None else [used_context("c1")])
    metadata = {"model": "fake", "context_count": len(contexts)}
    if prompt is True:
        metadata["prompt"] = valid_prompt(contexts)
    elif isinstance(prompt, str):
        metadata["prompt"] = prompt

    return {
        "answer": answer,
        "citations": list(citations if citations is not None else [citation("c1")]),
        "used_context": contexts,
        "insufficient_evidence": insufficient_evidence,
        "metadata": metadata,
    }


class MiniRagGroundednessTests(unittest.TestCase):
    def test_valid_answer_with_citation_inside_used_context_passes(self):
        validation = validate_groundedness(answer_result())

        self.assertTrue(validation["valid"])
        self.assertTrue(all(validation["checks"].values()))
        self.assertEqual(validation["unknown_citations"], [])
        self.assertEqual(validation["citations_outside_context"], [])
        self.assertEqual(validation["missing_citations"], [])
        self.assertEqual(validation["errors"], [])

    def test_citation_missing_chunk_id_fails(self):
        result = answer_result(citations=[{"source_chunk_id": "source-c1"}])

        validation = validate_groundedness(result)

        self.assertFalse(validation["valid"])
        self.assertFalse(validation["checks"]["citations_exist"])
        self.assertEqual(validation["missing_citations"], ["citations[0].chunk_id"])
        self.assertIn("citations_missing_chunk_id", validation["errors"])

    def test_citation_outside_used_context_fails(self):
        result = answer_result(citations=[citation("c2")])

        validation = validate_groundedness(result)

        self.assertFalse(validation["valid"])
        self.assertFalse(validation["checks"]["citations_in_used_context"])
        self.assertEqual(validation["citations_outside_context"], ["c2"])
        self.assertIn("citations_outside_used_context", validation["errors"])

    def test_answer_without_citations_fails_when_evidence_is_sufficient(self):
        result = answer_result(citations=[])

        validation = validate_groundedness(result)

        self.assertFalse(validation["valid"])
        self.assertFalse(validation["checks"]["citations_exist"])
        self.assertEqual(validation["missing_citations"], ["citations"])
        self.assertIn("citations_required_when_context_is_used", validation["errors"])

    def test_insufficient_evidence_with_standard_answer_passes(self):
        result = answer_result(
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            citations=[],
            contexts=[],
            insufficient_evidence=True,
            prompt=False,
        )

        validation = validate_groundedness(result)

        self.assertTrue(validation["valid"])
        self.assertTrue(validation["checks"]["insufficient_evidence_consistent"])
        self.assertIn("prompt_not_available_for_validation", validation["warnings"])

    def test_insufficient_evidence_with_citations_fails(self):
        result = answer_result(
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            citations=[citation("c1")],
            insufficient_evidence=True,
        )

        validation = validate_groundedness(result)

        self.assertFalse(validation["valid"])
        self.assertFalse(validation["checks"]["insufficient_evidence_consistent"])
        self.assertIn("insufficient_evidence_has_citations", validation["errors"])

    def test_empty_used_context_fails_when_evidence_is_sufficient(self):
        result = answer_result(contexts=[], prompt=False)

        validation = validate_groundedness(result)

        self.assertFalse(validation["valid"])
        self.assertFalse(validation["checks"]["used_context_consistent"])
        self.assertIn("used_context_empty_without_insufficient_evidence", validation["errors"])

    def test_answer_text_mentions_unknown_chunk(self):
        result = answer_result(answer="Resposta com fonte inexistente [c999] e fonte chunk_999.")

        validation = validate_groundedness(result)

        self.assertFalse(validation["valid"])
        self.assertEqual(validation["unknown_citations"], ["c999", "chunk_999"])
        self.assertIn("answer_mentions_unknown_chunk", validation["errors"])

    def test_prompt_with_groundedness_passes(self):
        validation = validate_groundedness(answer_result())

        self.assertTrue(validation["checks"]["prompt_contains_grounding"])
        self.assertTrue(validation["checks"]["prompt_contains_context"])
        self.assertTrue(validation["checks"]["prompt_contains_chunk_ids"])
        self.assertTrue(validation["checks"]["prompt_contains_pages"])
        self.assertTrue(validation["checks"]["prompt_contains_sections"])

    def test_missing_prompt_adds_warning_without_invalidating(self):
        result = answer_result(prompt=False)

        validation = validate_groundedness(result)

        self.assertTrue(validation["valid"])
        self.assertIn("prompt_not_available_for_validation", validation["warnings"])
        self.assertFalse(validation["checks"]["prompt_contains_grounding"])

    def test_prompt_without_chunk_ids_sets_check_false(self):
        prompt_without_chunk_ids = (
            "Você é um gerador de respostas RAG auditável.\n"
            "Responda somente usando os trechos fornecidos.\n"
            "Não use conhecimento externo.\n"
            f"Se não houver evidência suficiente, responda: {INSUFFICIENT_EVIDENCE_MESSAGE}\n"
            "Pergunta do usuário:\nQual é o objeto?\n"
            "Contexto recuperado:\n"
            "pages: 2-3\n"
            "section_path: [\"1 DO OBJETO\"]\n"
            "content: Conteúdo auditável."
        )
        result = answer_result(prompt=prompt_without_chunk_ids)

        validation = validate_groundedness(result)

        self.assertFalse(validation["valid"])
        self.assertFalse(validation["checks"]["prompt_contains_chunk_ids"])
        self.assertIn("prompt_contains_chunk_ids_missing", validation["errors"])

    def test_related_assets_and_source_spans_are_not_lost_during_validation(self):
        spans = [{"block_id": "b1", "chunk_start_char": 0, "chunk_end_char": 10}]
        assets = [{"asset_id": "asset1", "link_strategy": "nearby"}]
        result = answer_result(
            citations=[citation("c1", source_spans=spans, related_assets=assets)],
            contexts=[used_context("c1", source_spans=spans, related_assets=assets)],
        )
        original = copy.deepcopy(result)

        validation = GroundednessValidator().validate(result)

        self.assertTrue(validation["valid"])
        self.assertEqual(result, original)


if __name__ == "__main__":
    unittest.main()
