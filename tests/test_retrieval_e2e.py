import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build_index as build_index_cli
import evaluate_retrieval as evaluate_retrieval_cli
from mini_rag import indexing
from mini_rag.embeddings import EmbeddingProvider
from mini_rag.evaluation import evaluate_retrieval, load_evaluation_dataset
from mini_rag.indexing import build_document_index
from mini_rag.retrieval import Retriever
from mini_rag.store import LocalVectorStore
from settings import Settings


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "retrieval_eval"
PARSED_SECTIONS_FIXTURE = FIXTURE_DIR / "parsed_sections.jsonl"
DATASET_FIXTURE = FIXTURE_DIR / "dataset.json"


class DeterministicE2EEmbeddingProvider(EmbeddingProvider):
    model_name = "fake-deterministic-e2e"

    def __init__(self, model_name=None):
        if model_name is not None:
            self.model_name = model_name

    def embed(self, texts):
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text):
        if "E2E_QUERY_OBJETO" in text or "E2E_OBJETO" in text:
            return [1.0, 0.0, 0.0]
        if "E2E_QUERY_PRAZO" in text or "E2E_PRAZO" in text:
            return [0.0, 1.0, 0.0]
        if "E2E_QUERY_VALOR" in text or "E2E_VALOR" in text:
            return [0.0, 0.0, 1.0]
        return [0.0, 0.0, 0.0]


class RetrievalEndToEndTests(unittest.TestCase):
    def test_pipeline_from_parsed_sections_to_report_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "document_index.json"

            with patch(
                "mini_rag.indexing.build_hierarchical_chunks",
                wraps=indexing.build_hierarchical_chunks,
            ) as build_chunks:
                build_document_index(
                    PARSED_SECTIONS_FIXTURE,
                    index_path,
                    embedding_provider=DeterministicE2EEmbeddingProvider(),
                    settings=Settings(index_path=str(index_path)),
                    input_type="parsed_sections",
                )

            self.assertTrue(build_chunks.called)

            store = LocalVectorStore.load(index_path)
            self.assertGreaterEqual(len(store.items), 9)
            retriever = Retriever(
                store,
                DeterministicE2EEmbeddingProvider(),
                top_k=5,
                similarity_threshold=None,
            )

            report = evaluate_retrieval(
                load_evaluation_dataset(DATASET_FIXTURE),
                retriever,
                top_k=5,
                dataset_path=DATASET_FIXTURE,
                index_path=index_path,
                index_metadata={
                    "embedding_model": store.embedding_model,
                    "source_path": store.source_path,
                },
            )

        self.assertEqual(report["schema_version"], "1.0")
        self.assertEqual(report["dataset"]["name"], "mini-rag-e2e-smoke")
        self.assertEqual(report["dataset"]["case_count"], 3)
        self.assertEqual(report["index"]["embedding_model"], "fake-deterministic-e2e")
        self.assertEqual(report["config"]["top_k"], 5)

        metrics = report["metrics"]
        self.assertAlmostEqual(metrics["top1_hit"], 1 / 3)
        self.assertAlmostEqual(metrics["top3_hit"], 2 / 3)
        self.assertAlmostEqual(metrics["top5_hit"], 1.0)
        self.assertAlmostEqual(metrics["recall@1"], 1 / 3)
        self.assertAlmostEqual(metrics["recall@3"], 2 / 3)
        self.assertAlmostEqual(metrics["recall@5"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], (1.0 + (1 / 3) + (1 / 5)) / 3)

        self.assertEqual(
            report["retrieval_diagnostics"],
            {
                "total_questions": 3,
                "questions_with_no_hit": 0,
                "questions_hit_via_source_chunk_id": 3,
                "questions_hit_via_chunk_id": 0,
                "questions_hit_only_via_chunk_id": 0,
                "questions_with_multiple_expected_items": 0,
            },
        )
        self.assertEqual([case["first_relevant_rank"] for case in report["cases"]], [1, 3, 5])
        self.assertEqual([case["id"] for case in report["cases"]], ["objeto-top1", "prazo-top3", "valor-top5"])
        self.assertEqual(report["metrics"]["hits_by_section_path"]["1 DO OBJETO"]["top1_hits"], 1)
        self.assertEqual(report["metrics"]["hits_by_section_path"]["2 DO PRAZO"]["top3_hits"], 1)
        self.assertEqual(report["metrics"]["hits_by_section_path"]["3 DO VALOR"]["top5_hits"], 1)
        self.assertEqual(report["metrics"]["hits_by_page"]["1"]["top1_hits"], 1)
        self.assertEqual(report["metrics"]["hits_by_page"]["2"]["top3_hits"], 1)
        self.assertEqual(report["metrics"]["hits_by_page"]["3"]["top5_hits"], 1)
        self.assertEqual(report["cases"][0]["results"][0]["source_chunk_id"], "parsed-objeto")

    def test_cli_build_index_then_evaluate_retrieval_writes_valid_report_and_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            index_path = tmp_path / "index" / "document_index.json"
            output_path = tmp_path / "reports" / "retrieval_eval_report.json"

            build_stdout = io.StringIO()
            with contextlib.redirect_stdout(build_stdout):
                build_exit_code = build_index_cli.main(
                    [
                        "--input",
                        str(PARSED_SECTIONS_FIXTURE),
                        "--input-type",
                        "parsed_sections",
                        "--output",
                        str(index_path),
                        "--model",
                        "fake-deterministic-e2e",
                    ],
                    embedding_provider_factory=DeterministicE2EEmbeddingProvider,
                )

            evaluate_stdout = io.StringIO()
            with contextlib.redirect_stdout(evaluate_stdout):
                evaluate_exit_code = evaluate_retrieval_cli.main(
                    [
                        "--dataset",
                        str(DATASET_FIXTURE),
                        "--index",
                        str(index_path),
                        "--output",
                        str(output_path),
                        "--top-k",
                        "5",
                    ],
                    embedding_provider_factory=DeterministicE2EEmbeddingProvider,
                )

            report = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(build_exit_code, 0)
            self.assertEqual(evaluate_exit_code, 0)
            self.assertTrue(index_path.exists())
            self.assertTrue(output_path.exists())
            self.assertEqual(report["metrics"]["top5_hit"], 1.0)
            self.assertEqual(report["cases"][2]["first_relevant_rank"], 5)
            self.assertIn("Wrote", build_stdout.getvalue())
            self.assertIn("vectors", build_stdout.getvalue())
            self.assertIn("dataset: mini-rag-e2e-smoke", evaluate_stdout.getvalue())
            self.assertIn("top1/top3/top5:", evaluate_stdout.getvalue())
            self.assertIn("output:", evaluate_stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
