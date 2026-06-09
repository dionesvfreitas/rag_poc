import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import evaluate_retrieval as evaluate_retrieval_cli
from mini_rag.evaluation import (
    evaluate_retrieval,
    evaluate_retrieval_case,
    expected_item_ids,
    expected_match_basis,
    load_evaluation_dataset,
    match_result,
    summarize_metrics,
)
from mini_rag.embeddings import EmbeddingProvider
from mini_rag.store import LocalVectorStore


class SyntheticRetriever:
    def __init__(self, results_by_question):
        self.results_by_question = results_by_question

    def search(self, query, top_k=None):
        results = list(self.results_by_question.get(query, []))
        return results[:top_k] if top_k is not None else results


class FakeCliEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name="fake-cli-embedding"):
        self.model_name = model_name

    def embed(self, texts):
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text):
        text = text.lower()
        if "alpha" in text:
            return [1.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0]
        return [0.0, 0.0]


class RecordingEmbeddingProviderFactory:
    def __init__(self):
        self.model_names = []

    def __call__(self, model_name):
        self.model_names.append(model_name)
        return FakeCliEmbeddingProvider(model_name)


def result(
    chunk_id,
    source_chunk_id=None,
    score=1.0,
    *,
    section_path=None,
    pages=None,
    content="",
    content_preview=None,
):
    chunk = {
        "chunk_id": chunk_id,
        "content": content,
    }
    if content_preview is not None:
        chunk["content_preview"] = content_preview
    metadata = {
        "source_chunk_id": source_chunk_id,
        "section_path": section_path if section_path is not None else ["ignored"],
        "page_start": 1,
    }
    if pages is not None:
        metadata["pages"] = pages
    return {
        "score": score,
        "chunk": chunk,
        "metadata": metadata,
    }


def dataset_case(case_id, question, expected):
    return {
        "id": case_id,
        "question": question,
        "expected": expected,
    }


def dataset(cases):
    return {
        "schema_version": "1.0",
        "name": "synthetic-retrieval-eval",
        "description": "Synthetic test dataset.",
        "default_top_k": 5,
        "cases": cases,
    }


def dataset_without_default_top_k(cases):
    return {
        "schema_version": "1.0",
        "name": "synthetic-retrieval-eval",
        "description": "Synthetic test dataset.",
        "cases": cases,
    }


class RetrievalEvaluationTests(unittest.TestCase):
    def test_source_chunk_id_is_prioritized(self):
        expected = {
            "source_chunk_ids": ["source-right"],
            "chunk_ids": ["chunk-right"],
            "section_paths": [["ignored"]],
            "pages": [1],
        }

        self.assertEqual(expected_match_basis(expected), "source_chunk_id")

        match = match_result(result("chunk-right", "source-wrong"), expected)

        self.assertFalse(match["matched"])
        self.assertEqual(match["match_type"], "no_match")

    def test_relevance_falls_back_to_chunk_id(self):
        expected = {
            "chunk_ids": ["chunk-right"],
            "section_paths": [["ignored"]],
            "pages": [1],
        }

        self.assertEqual(expected_match_basis(expected), "chunk_id")

        match = match_result(result("chunk-right", "source-any"), expected)

        self.assertTrue(match["matched"])
        self.assertEqual(match["match_type"], "chunk_id")

    def test_match_type_is_reported_for_source_chunk_id_chunk_id_and_no_match(self):
        source_match = match_result(
            result("chunk-any", "source-right"),
            {"source_chunk_ids": ["source-right"], "chunk_ids": ["chunk-any"]},
        )
        chunk_match = match_result(
            result("chunk-right", "source-any"),
            {"source_chunk_ids": [], "chunk_ids": ["chunk-right"]},
        )
        no_match = match_result(
            result("chunk-wrong", "source-wrong"),
            {"source_chunk_ids": ["source-right"], "chunk_ids": ["chunk-wrong"]},
        )

        self.assertEqual(source_match["match_type"], "source_chunk_id")
        self.assertEqual(chunk_match["match_type"], "chunk_id")
        self.assertEqual(no_match["match_type"], "no_match")

    def test_summarize_metrics_computes_topk_and_mrr(self):
        cases = [
            evaluate_retrieval_case(
                dataset_case("rank-1", "q1", {"source_chunk_ids": ["s1"]}),
                [result("c1", "s1")],
            ),
            evaluate_retrieval_case(
                dataset_case("rank-2", "q2", {"source_chunk_ids": ["s2"]}),
                [result("cx", "sx"), result("c2", "s2")],
            ),
            evaluate_retrieval_case(
                dataset_case("rank-4", "q3", {"source_chunk_ids": ["s3"]}),
                [
                    result("cx1", "sx1"),
                    result("cx2", "sx2"),
                    result("cx3", "sx3"),
                    result("c3", "s3"),
                ],
            ),
        ]

        metrics = summarize_metrics(cases)

        self.assertAlmostEqual(metrics["top1_hit"], 1 / 3)
        self.assertAlmostEqual(metrics["top3_hit"], 2 / 3)
        self.assertAlmostEqual(metrics["top5_hit"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], (1.0 + 0.5 + 0.25) / 3)

    def test_recall_at_1_3_and_5_uses_active_basis(self):
        case = evaluate_retrieval_case(
            dataset_case("multi", "q", {"source_chunk_ids": ["s1", "s2", "s3"]}),
            [
                result("c1", "s1"),
                result("cx", "sx"),
                result("c2", "s2"),
                result("cy", "sy"),
                result("c3", "s3"),
            ],
        )

        self.assertEqual(case["retrieved_expected_count@1"], 1)
        self.assertEqual(case["retrieved_expected_count@3"], 2)
        self.assertEqual(case["retrieved_expected_count@5"], 3)
        self.assertAlmostEqual(case["recall@1"], 1 / 3)
        self.assertAlmostEqual(case["recall@3"], 2 / 3)
        self.assertAlmostEqual(case["recall@5"], 1.0)

    def test_recall_deduplicates_expected_and_retrieved_ids(self):
        case = evaluate_retrieval_case(
            dataset_case("dedupe", "q", {"source_chunk_ids": ["s1", "s1", "s2"]}),
            [
                result("c1", "s1"),
                result("c1-duplicate", "s1"),
                result("c2", "s2"),
            ],
        )

        self.assertEqual(expected_item_ids({"source_chunk_ids": ["s1", "s1", "s2"]}), ["s1", "s2"])
        self.assertEqual(case["expected_count"], 2)
        self.assertEqual(case["retrieved_expected_count@1"], 1)
        self.assertEqual(case["retrieved_expected_count@3"], 2)
        self.assertAlmostEqual(case["recall@1"], 0.5)
        self.assertAlmostEqual(case["recall@3"], 1.0)
        self.assertAlmostEqual(case["recall@5"], 1.0)

    def test_invalid_dataset_without_expected_ids_raises_clear_error(self):
        invalid_dataset = dataset(
            [
                dataset_case(
                    "invalid",
                    "q",
                    {"section_paths": [["section-only"]], "pages": [2]},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.json"
            path.write_text(json.dumps(invalid_dataset), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must define expected.source_chunk_ids or expected.chunk_ids"):
                load_evaluation_dataset(path)

    def test_case_without_hit_returns_zero_metrics(self):
        case = evaluate_retrieval_case(
            dataset_case("miss", "q", {"source_chunk_ids": ["s1"]}),
            [
                result("cx", "sx"),
                result("cy", "sy"),
            ],
        )

        self.assertIsNone(case["first_relevant_rank"])
        self.assertEqual(case["reciprocal_rank"], 0.0)
        self.assertEqual(case["first_hit_match_type"], "no_match")
        self.assertFalse(case["top1_hit"])
        self.assertFalse(case["top3_hit"])
        self.assertFalse(case["top5_hit"])
        self.assertEqual(case["recall@1"], 0.0)
        self.assertEqual(case["recall@3"], 0.0)
        self.assertEqual(case["recall@5"], 0.0)

    def test_evaluate_retrieval_runs_questions_with_synthetic_retriever(self):
        evaluation_dataset = dataset(
            [
                dataset_case("hit", "q1", {"source_chunk_ids": ["s1"]}),
                dataset_case("fallback", "q2", {"chunk_ids": ["c2"]}),
            ]
        )
        retriever = SyntheticRetriever(
            {
                "q1": [result("c1", "s1")],
                "q2": [result("c2", None)],
            }
        )

        evaluation = evaluate_retrieval(evaluation_dataset, retriever)

        self.assertEqual([case["id"] for case in evaluation["cases"]], ["hit", "fallback"])
        self.assertEqual(evaluation["cases"][0]["first_hit_match_type"], "source_chunk_id")
        self.assertEqual(evaluation["cases"][1]["first_hit_match_type"], "chunk_id")
        self.assertEqual(evaluation["metrics"]["top1_hit"], 1.0)

    def test_report_contains_metadata_metrics_diagnostics_and_cases(self):
        evaluation_dataset = dataset(
            [
                dataset_case(
                    "hit",
                    "q1",
                    {
                        "source_chunk_ids": ["s1"],
                        "section_paths": [["Section A"]],
                        "pages": [2],
                    },
                )
            ]
        )
        retriever = SyntheticRetriever(
            {
                "q1": [
                    result(
                        "c1",
                        "s1",
                        section_path=["Section A"],
                        pages=[2],
                    )
                ]
            }
        )

        report = evaluate_retrieval(
            evaluation_dataset,
            retriever,
            dataset_path="tests/fixtures/retrieval_eval/dataset.json",
            index_path="index/document_index.json",
            index_metadata={
                "embedding_model": "fake-keyword-embedding",
                "source_path": "parsed_sections.jsonl",
            },
            top_k=5,
        )

        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(
            report["dataset"],
            {
                "name": "synthetic-retrieval-eval",
                "path": "tests/fixtures/retrieval_eval/dataset.json",
                "case_count": 1,
            },
        )
        self.assertEqual(report["index"]["path"], "index/document_index.json")
        self.assertEqual(report["index"]["embedding_model"], "fake-keyword-embedding")
        self.assertEqual(report["index"]["source_path"], "parsed_sections.jsonl")
        self.assertEqual(report["config"], {"top_k": 5, "similarity_threshold": None})
        self.assertIn("metrics", report)
        self.assertIn("hits_by_section_path", report["metrics"])
        self.assertIn("hits_by_page", report["metrics"])
        self.assertIn("retrieval_diagnostics", report)
        self.assertEqual(len(report["cases"]), 1)

    def test_retrieval_diagnostics_counts_hit_types_no_hit_and_multiple_expected_items(self):
        evaluation_dataset = dataset(
            [
                dataset_case("source-hit", "q1", {"source_chunk_ids": ["s1"]}),
                dataset_case("chunk-hit", "q2", {"source_chunk_ids": [], "chunk_ids": ["c2"]}),
                dataset_case("miss", "q3", {"source_chunk_ids": ["s3"]}),
                dataset_case("multi", "q4", {"source_chunk_ids": ["s4", "s5"]}),
            ]
        )
        retriever = SyntheticRetriever(
            {
                "q1": [result("c1", "s1")],
                "q2": [result("c2", None)],
                "q3": [result("wrong", "wrong")],
                "q4": [result("wrong", "wrong"), result("c5", "s5")],
            }
        )

        report = evaluate_retrieval(evaluation_dataset, retriever)

        self.assertEqual(
            report["retrieval_diagnostics"],
            {
                "total_questions": 4,
                "questions_with_no_hit": 1,
                "questions_hit_via_source_chunk_id": 2,
                "questions_hit_via_chunk_id": 1,
                "questions_hit_only_via_chunk_id": 1,
                "questions_with_multiple_expected_items": 1,
            },
        )

    def test_hits_by_section_path_does_not_create_false_hit(self):
        evaluation_dataset = dataset(
            [
                dataset_case(
                    "section-only-match",
                    "q",
                    {
                        "source_chunk_ids": ["s1"],
                        "section_paths": [["Expected", "Section"]],
                    },
                )
            ]
        )
        retriever = SyntheticRetriever(
            {
                "q": [
                    result(
                        "wrong",
                        "wrong",
                        section_path=["Expected", "Section"],
                    )
                ]
            }
        )

        report = evaluate_retrieval(evaluation_dataset, retriever)

        self.assertIsNone(report["cases"][0]["first_relevant_rank"])
        self.assertEqual(report["metrics"]["hits_by_section_path"], {})

    def test_hits_by_page_does_not_create_false_hit(self):
        evaluation_dataset = dataset(
            [
                dataset_case(
                    "page-only-match",
                    "q",
                    {
                        "source_chunk_ids": ["s1"],
                        "pages": [7],
                    },
                )
            ]
        )
        retriever = SyntheticRetriever({"q": [result("wrong", "wrong", pages=[7])]})

        report = evaluate_retrieval(evaluation_dataset, retriever)

        self.assertIsNone(report["cases"][0]["first_relevant_rank"])
        self.assertEqual(report["metrics"]["hits_by_page"], {})

    def test_section_and_page_aggregate_only_cases_with_primary_hit(self):
        evaluation_dataset = dataset(
            [
                dataset_case(
                    "hit",
                    "q1",
                    {
                        "source_chunk_ids": ["s1"],
                        "section_paths": [["Same", "Section"]],
                        "pages": [3],
                    },
                ),
                dataset_case(
                    "miss",
                    "q2",
                    {
                        "source_chunk_ids": ["s2"],
                        "section_paths": [["Same", "Section"]],
                        "pages": [3],
                    },
                ),
            ]
        )
        retriever = SyntheticRetriever(
            {
                "q1": [result("c1", "s1")],
                "q2": [result("wrong", "wrong")],
            }
        )

        report = evaluate_retrieval(evaluation_dataset, retriever)

        section_bucket = report["metrics"]["hits_by_section_path"]["Same > Section"]
        page_bucket = report["metrics"]["hits_by_page"]["3"]
        self.assertEqual(section_bucket["expected_cases"], 1)
        self.assertEqual(section_bucket["top1_hits"], 1)
        self.assertEqual(section_bucket["recall@1"], 1.0)
        self.assertEqual(page_bucket["expected_cases"], 1)
        self.assertEqual(page_bucket["top1_hits"], 1)
        self.assertEqual(page_bucket["recall@1"], 1.0)

    def test_mrr_and_first_relevant_rank_ignore_hit_outside_top_k(self):
        case = evaluate_retrieval_case(
            dataset_case("outside-top-k", "q", {"source_chunk_ids": ["s6"]}),
            [
                result("c1", "s1"),
                result("c2", "s2"),
                result("c3", "s3"),
                result("c4", "s4"),
                result("c5", "s5"),
                result("c6", "s6"),
            ],
            top_k=5,
        )

        self.assertIsNone(case["first_relevant_rank"])
        self.assertEqual(case["reciprocal_rank"], 0.0)
        self.assertFalse(case["top1_hit"])
        self.assertFalse(case["top3_hit"])
        self.assertFalse(case["top5_hit"])
        self.assertEqual(case["recall@1"], 0.0)
        self.assertEqual(case["recall@3"], 0.0)
        self.assertEqual(case["recall@5"], 0.0)
        self.assertEqual(len(case["results"]), 5)

        metrics = summarize_metrics([case])
        self.assertEqual(metrics["mrr"], 0.0)

    def test_results_include_match_fields_location_and_no_full_content(self):
        long_preview = "x" * 250
        long_content = "full content should stay out of the report"
        case = evaluate_retrieval_case(
            dataset_case("hit", "q", {"source_chunk_ids": ["s1"]}),
            [
                result(
                    "c1",
                    "s1",
                    section_path=["Section"],
                    pages=[2],
                    content=long_content,
                    content_preview=long_preview,
                )
            ],
        )

        reported_result = case["results"][0]
        self.assertTrue(reported_result["matched"])
        self.assertEqual(reported_result["match_type"], "source_chunk_id")
        self.assertEqual(reported_result["section_path"], ["Section"])
        self.assertEqual(reported_result["pages"], [2])
        self.assertNotIn("content", reported_result)
        self.assertEqual(len(reported_result["content_preview"]), 200)

    def test_top_k_less_than_5_raises_clear_core_error(self):
        with self.assertRaisesRegex(ValueError, "top_k must be >= 5"):
            evaluate_retrieval_case(
                dataset_case("too-small", "q", {"source_chunk_ids": ["s1"]}),
                [result("c1", "s1")],
                top_k=3,
            )

    def test_cli_writes_report_creates_output_dir_uses_fake_provider_and_prints_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "dataset.json"
            index_path = tmp_path / "document_index.json"
            output_path = tmp_path / "reports" / "nested" / "retrieval_eval_report.json"
            dataset_path.write_text(
                json.dumps(
                    dataset(
                        [
                            dataset_case(
                                "alpha-hit",
                                "alpha question",
                                {
                                    "source_chunk_ids": ["source-alpha"],
                                    "section_paths": [["Section Alpha"]],
                                    "pages": [1],
                                },
                            )
                        ]
                    ),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            store = LocalVectorStore(
                embedding_model="fake-index-model",
                source_path=None,
            )
            store.add(
                "chunk-alpha",
                "alpha document",
                [1.0, 0.0],
                {"source_chunk_id": "source-alpha", "section_path": ["Section Alpha"], "pages": [1]},
            )
            store.add(
                "chunk-beta",
                "beta document",
                [0.0, 1.0],
                {"source_chunk_id": "source-beta", "section_path": ["Section Beta"], "pages": [2]},
            )
            store.save(index_path)
            provider_factory = RecordingEmbeddingProviderFactory()

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = evaluate_retrieval_cli.main(
                    [
                        "--dataset",
                        str(dataset_path),
                        "--index",
                        str(index_path),
                        "--output",
                        str(output_path),
                        "--top-k",
                        "5",
                        "--threshold",
                        "0.25",
                    ],
                    embedding_provider_factory=provider_factory,
                )

            report = json.loads(output_path.read_text(encoding="utf-8"))
            printed_summary = stdout.getvalue()

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            self.assertEqual(provider_factory.model_names, ["fake-index-model"])
            self.assertEqual(set(report), {"schema_version", "dataset", "index", "config", "metrics", "retrieval_diagnostics", "cases"})
            self.assertEqual(report["dataset"]["name"], "synthetic-retrieval-eval")
            self.assertEqual(report["dataset"]["case_count"], 1)
            self.assertEqual(report["index"]["path"], str(index_path))
            self.assertEqual(report["index"]["embedding_model"], "fake-index-model")
            self.assertIsNone(report["index"]["source_path"])
            self.assertEqual(report["config"]["top_k"], 5)
            self.assertEqual(report["config"]["similarity_threshold"], 0.25)
            self.assertEqual(report["metrics"]["top1_hit"], 1.0)
            self.assertEqual(report["metrics"]["recall@5"], 1.0)
            self.assertEqual(report["cases"][0]["id"], "alpha-hit")
            self.assertIn("dataset: synthetic-retrieval-eval", printed_summary)
            self.assertIn("case_count: 1", printed_summary)
            self.assertIn("top1/top3/top5:", printed_summary)
            self.assertIn("recall@1/@3/@5:", printed_summary)
            self.assertIn("mrr:", printed_summary)
            self.assertIn(f"output: {output_path}", printed_summary)

    def test_cli_top_k_3_fails_with_clear_message_before_loading_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "dataset.json"
            index_path = tmp_path / "document_index.json"
            output_path = tmp_path / "report.json"
            dataset_path.write_text(
                json.dumps(
                    dataset([dataset_case("alpha-hit", "alpha question", {"source_chunk_ids": ["source-alpha"]})])
                ),
                encoding="utf-8",
            )
            LocalVectorStore(embedding_model="fake-index-model").save(index_path)
            provider_factory = RecordingEmbeddingProviderFactory()

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    evaluate_retrieval_cli.main(
                        [
                            "--dataset",
                            str(dataset_path),
                            "--index",
                            str(index_path),
                            "--output",
                            str(output_path),
                            "--top-k",
                            "3",
                        ],
                        embedding_provider_factory=provider_factory,
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("top_k must be >= 5", stderr.getvalue())
            self.assertEqual(provider_factory.model_names, [])
            self.assertFalse(output_path.exists())

    def test_cli_uses_dataset_default_top_k_when_present(self):
        report = self._run_cli_and_read_report(
            dataset(
                [
                    dataset_case(
                        "alpha-hit",
                        "alpha question",
                        {"source_chunk_ids": ["source-alpha"]},
                    )
                ]
            )
            | {"default_top_k": 6}
        )

        self.assertEqual(report["config"]["top_k"], 6)

    def test_cli_defaults_top_k_to_5_when_dataset_does_not_define_it(self):
        report = self._run_cli_and_read_report(
            dataset_without_default_top_k(
                [
                    dataset_case(
                        "alpha-hit",
                        "alpha question",
                        {"source_chunk_ids": ["source-alpha"]},
                    )
                ]
            )
        )

        self.assertEqual(report["config"]["top_k"], 5)

    def _run_cli_and_read_report(self, evaluation_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "dataset.json"
            index_path = tmp_path / "document_index.json"
            output_path = tmp_path / "report.json"
            dataset_path.write_text(
                json.dumps(evaluation_dataset, ensure_ascii=False),
                encoding="utf-8",
            )
            store = LocalVectorStore(embedding_model="fake-index-model", source_path="synthetic.jsonl")
            store.add(
                "chunk-alpha",
                "alpha document",
                [1.0, 0.0],
                {"source_chunk_id": "source-alpha"},
            )
            store.save(index_path)

            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_retrieval_cli.main(
                    [
                        "--dataset",
                        str(dataset_path),
                        "--index",
                        str(index_path),
                        "--output",
                        str(output_path),
                    ],
                    embedding_provider_factory=RecordingEmbeddingProviderFactory(),
                )

            return json.loads(output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
