import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


class ReadmeOperationalTests(unittest.TestCase):
    def test_readme_documents_existing_operational_commands(self):
        text = README.read_text(encoding="utf-8")

        expected_commands = [
            ".venv/bin/python parse_pdf.py",
            ".venv/bin/python rag_chunker.py",
            ".venv/bin/python build_index.py --input parsed_sections.jsonl --input-type parsed_sections --output index/document_index.json",
            ".venv/bin/python build_index.py --input tests/fixtures/retrieval_eval/parsed_sections.jsonl --input-type parsed_sections --output index/document_index.json",
            ".venv/bin/python evaluate_retrieval.py --dataset tests/fixtures/retrieval_eval/dataset.json --index index/document_index.json --output reports/retrieval_eval_report.json --top-k 5",
            ".venv/bin/python -m unittest discover -s tests -v",
        ]
        for command in expected_commands:
            self.assertIn(command, _single_line_commands(text))

        for script in ["parse_pdf.py", "rag_chunker.py", "build_index.py", "evaluate_retrieval.py"]:
            self.assertTrue((ROOT / script).exists(), script)
        self.assertTrue((ROOT / "tests" / "fixtures" / "retrieval_eval" / "dataset.json").exists())

    def test_readme_explains_retrieval_metrics_and_current_limitations(self):
        text = README.read_text(encoding="utf-8")

        for metric in ["top1_hit", "top3_hit", "top5_hit", "MRR", "recall@1", "recall@3", "recall@5"]:
            self.assertIn(metric, text)

        for limitation in ["sem BM25", "sem reranker", "sem vector DB externo", "sem LLM", "sem API"]:
            self.assertIn(limitation, text)

    def test_readme_does_not_document_nonexistent_integrations_as_available(self):
        text = README.read_text(encoding="utf-8")

        forbidden_patterns = [
            r"pip install .*qdrant",
            r"pip install .*chromadb",
            r"pip install .*elasticsearch",
            r"\bollama\b",
            r"\bfastapi\b",
            r"uvicorn .*app",
            r"docker run .*qdrant",
            r"docker run .*chroma",
        ]
        for pattern in forbidden_patterns:
            self.assertIsNone(re.search(pattern, text, flags=re.IGNORECASE), pattern)


def _single_line_commands(markdown_text):
    commands = set()
    in_code_block = False
    current = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            if in_code_block and current:
                commands.add(" ".join(part.rstrip("\\").strip() for part in current if part.strip()))
            in_code_block = not in_code_block
            current = []
            continue
        if not in_code_block:
            continue
        if not line.strip():
            if current:
                commands.add(" ".join(part.rstrip("\\").strip() for part in current if part.strip()))
                current = []
            continue
        if line.endswith("\\"):
            current.append(line)
            continue
        current.append(line)
        commands.add(" ".join(part.rstrip("\\").strip() for part in current if part.strip()))
        current = []
    return commands


if __name__ == "__main__":
    unittest.main()
