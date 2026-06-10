import unittest

from mini_rag.answering import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    AnswerGenerator,
    FakeLLMProvider,
    LLMProvider,
    build_answer_prompt,
)


def result(
    chunk_id,
    score=0.9,
    *,
    content=None,
    source_chunk_id=None,
    page_start=2,
    page_end=3,
    section_path=None,
    source_spans=None,
    source_block_ids=None,
    related_assets=None,
):
    return {
        "score": score,
        "chunk": {"chunk_id": chunk_id, "content": content or f"Conteúdo do {chunk_id}."},
        "metadata": {
            "source_chunk_id": source_chunk_id or f"source-{chunk_id}",
            "page_start": page_start,
            "page_end": page_end,
            "section_path": list(section_path or ["1 DO OBJETO"]),
            "source_spans": list(source_spans or []),
            "source_block_ids": list(source_block_ids or []),
            "related_assets": list(related_assets or []),
        },
    }


class ExplodingLLMProvider:
    model = "exploding"

    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        del prompt
        self.call_count += 1
        raise AssertionError("LLM should not be called")


class MiniRagAnsweringTests(unittest.TestCase):
    def test_llm_provider_protocol_is_importable(self):
        self.assertTrue(hasattr(LLMProvider, "generate"))

    def test_prompt_contains_question_context_ids_pages_sections_and_rules(self):
        fake = FakeLLMProvider()
        generator = AnswerGenerator(fake)
        generator.answer(
            "Qual é o objeto?",
            [
                result(
                    "c1",
                    content="O objeto é contratar solução inovadora.",
                    page_start=5,
                    page_end=6,
                    section_path=["1 DO OBJETO", "1.1"],
                )
            ],
        )

        prompt = fake.prompts[0]
        self.assertIn("Qual é o objeto?", prompt)
        self.assertIn("O objeto é contratar solução inovadora.", prompt)
        self.assertIn("chunk_id: c1", prompt)
        self.assertIn("pages: 5-6", prompt)
        self.assertIn('"1 DO OBJETO"', prompt)
        self.assertIn("Responda somente usando os trechos fornecidos.", prompt)
        self.assertIn("Não use conhecimento externo.", prompt)
        self.assertIn(INSUFFICIENT_EVIDENCE_MESSAGE, prompt)
        self.assertIn("Cite apenas fontes fornecidas", prompt)

    def test_build_answer_prompt_can_be_tested_directly(self):
        prompt = build_answer_prompt(
            "Pergunta?",
            [
                {
                    "rank": 1,
                    "score": 0.8,
                    "chunk_id": "c1",
                    "source_chunk_id": "s1",
                    "page_start": 1,
                    "page_end": 1,
                    "section_path": ["A"],
                    "source_block_ids": [],
                    "source_spans": [],
                    "related_assets": [],
                    "content": "Texto.",
                }
            ],
        )

        self.assertIn("Pergunta?", prompt)
        self.assertIn("Texto.", prompt)
        self.assertIn("chunk_id: c1", prompt)

    def test_fake_llm_generates_deterministic_grounded_answer(self):
        fake = FakeLLMProvider()
        response = AnswerGenerator(fake).answer("Pergunta?", [result("c1")])

        self.assertEqual(response["answer"], "Resposta baseada nos trechos recuperados. [c1]")
        self.assertEqual(fake.call_count, 1)
        self.assertFalse(response["insufficient_evidence"])
        self.assertEqual(response["metadata"]["model"], "fake")

    def test_fake_llm_can_generate_insufficient_answer(self):
        fake = FakeLLMProvider(insufficient=True)
        response = AnswerGenerator(fake).answer("Pergunta?", [result("c1")])

        self.assertEqual(response["answer"], INSUFFICIENT_EVIDENCE_MESSAGE)
        self.assertTrue(response["insufficient_evidence"])

    def test_result_includes_citations_and_preserves_audit_fields(self):
        spans = [{"block_id": "b1", "chunk_start_char": 0, "chunk_end_char": 10}]
        assets = [{"asset_id": "fig1", "link_strategy": "nearby"}]
        response = AnswerGenerator(FakeLLMProvider()).answer(
            "Pergunta?",
            [
                result(
                    "c1",
                    score=0.77,
                    source_chunk_id="source-1",
                    source_spans=spans,
                    source_block_ids=["b1", "b2"],
                    related_assets=assets,
                )
            ],
        )

        self.assertEqual(
            response["citations"],
            [
                {
                    "chunk_id": "c1",
                    "source_chunk_id": "source-1",
                    "pages": [2, 3],
                    "page_start": 2,
                    "page_end": 3,
                    "section_path": ["1 DO OBJETO"],
                    "source_block_ids": ["b1", "b2"],
                    "source_spans": spans,
                    "related_assets": assets,
                    "score": 0.77,
                }
            ],
        )

    def test_no_results_returns_insufficient_evidence_without_calling_llm(self):
        llm = ExplodingLLMProvider()
        response = AnswerGenerator(llm).answer("Pergunta?", [])

        self.assertEqual(response["answer"], INSUFFICIENT_EVIDENCE_MESSAGE)
        self.assertTrue(response["insufficient_evidence"])
        self.assertEqual(response["citations"], [])
        self.assertEqual(response["used_context"], [])
        self.assertEqual(llm.call_count, 0)

    def test_scores_below_threshold_return_insufficient_without_calling_llm(self):
        llm = ExplodingLLMProvider()
        response = AnswerGenerator(llm, evidence_score_threshold=0.5).answer(
            "Pergunta?",
            [result("c1", score=0.1), result("c2", score=0.49)],
        )

        self.assertTrue(response["insufficient_evidence"])
        self.assertEqual(response["citations"], [])
        self.assertEqual(llm.call_count, 0)

    def test_score_threshold_calls_llm_when_any_result_meets_threshold(self):
        fake = FakeLLMProvider()
        response = AnswerGenerator(fake, evidence_score_threshold=0.5).answer(
            "Pergunta?",
            [result("c1", score=0.1), result("c2", score=0.5)],
        )

        self.assertFalse(response["insufficient_evidence"])
        self.assertEqual(fake.call_count, 1)

    def test_no_real_model_is_used_in_unit_tests(self):
        fake = FakeLLMProvider(response="Resposta fixa. [c1]")
        response = AnswerGenerator(fake).answer("Pergunta?", [result("c1")])

        self.assertEqual(response["answer"], "Resposta fixa. [c1]")
        self.assertEqual(fake.call_count, 1)
        self.assertEqual(response["metadata"]["model"], "fake")

    def test_answer_removes_unknown_explicit_chunk_references(self):
        fake = FakeLLMProvider(response="Resposta apoiada. [c1] [c-inexistente]")
        response = AnswerGenerator(fake).answer("Pergunta?", [result("c1")])

        self.assertIn("[c1]", response["answer"])
        self.assertNotIn("c-inexistente", response["answer"])

    def test_used_context_contains_only_chunks_sent_to_prompt(self):
        fake = FakeLLMProvider()
        response = AnswerGenerator(fake).answer(
            "Pergunta?",
            [result("c1", content="Primeiro."), result("c2", content="Segundo.")],
        )

        self.assertEqual([context["chunk_id"] for context in response["used_context"]], ["c1", "c2"])
        self.assertIn("chunk_id: c1", fake.prompts[0])
        self.assertIn("chunk_id: c2", fake.prompts[0])
        self.assertNotIn("chunk_id: c3", fake.prompts[0])

    def test_used_context_includes_required_context_fields(self):
        spans = [{"block_id": "b1"}]
        assets = [{"asset_id": "asset1"}]
        response = AnswerGenerator(FakeLLMProvider()).answer(
            "Pergunta?",
            [
                result(
                    "c1",
                    source_spans=spans,
                    source_block_ids=["b1"],
                    related_assets=assets,
                )
            ],
        )

        context = response["used_context"][0]
        self.assertEqual(context["rank"], 1)
        self.assertEqual(context["score"], 0.9)
        self.assertEqual(context["chunk_id"], "c1")
        self.assertEqual(context["source_chunk_id"], "source-c1")
        self.assertEqual(context["page_start"], 2)
        self.assertEqual(context["page_end"], 3)
        self.assertEqual(context["section_path"], ["1 DO OBJETO"])
        self.assertEqual(context["source_spans"], spans)
        self.assertEqual(context["source_block_ids"], ["b1"])
        self.assertEqual(context["related_assets"], assets)

    def test_related_assets_are_included_in_prompt_when_present(self):
        fake = FakeLLMProvider()
        AnswerGenerator(fake).answer(
            "Pergunta?",
            [result("c1", related_assets=[{"asset_id": "fig1"}])],
        )

        self.assertIn("related_assets:", fake.prompts[0])
        self.assertIn("fig1", fake.prompts[0])


if __name__ == "__main__":
    unittest.main()
