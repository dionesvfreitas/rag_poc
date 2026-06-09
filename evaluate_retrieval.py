import argparse
import json
from pathlib import Path
from typing import Callable

from mini_rag.embeddings import EmbeddingProvider, SentenceTransformersEmbeddingProvider
from mini_rag.evaluation import evaluate_retrieval, load_evaluation_dataset, validate_top_k
from mini_rag.retrieval import Retriever
from mini_rag.store import LocalVectorStore
from settings import Settings


EmbeddingProviderFactory = Callable[[str], EmbeddingProvider]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate local Mini-RAG retrieval quality.")
    parser.add_argument("--dataset", required=True, help="Path to retrieval evaluation dataset JSON.")
    parser.add_argument("--index", required=True, help="Path to document_index.json.")
    parser.add_argument("--output", required=True, help="Path where the report JSON will be written.")
    parser.add_argument("--top-k", type=int, default=None, help="Largest top-k to evaluate. Must be >= 5.")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum cosine similarity score.")
    parser.add_argument("--model", default=None, help="Override embedding model from the index.")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def default_embedding_provider_factory(model_name: str) -> EmbeddingProvider:
    return SentenceTransformersEmbeddingProvider(model_name)


def main(argv=None, embedding_provider_factory: EmbeddingProviderFactory | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset = load_evaluation_dataset(args.dataset)
    try:
        top_k = validate_top_k(args.top_k if args.top_k is not None else dataset.get("default_top_k", 5))
    except ValueError as exc:
        parser.error(str(exc))

    store = LocalVectorStore.load(args.index)
    model_name = args.model or store.embedding_model or Settings.from_env().embedding_model
    provider_factory = embedding_provider_factory or default_embedding_provider_factory
    embedding_provider = provider_factory(model_name)
    retriever = Retriever(
        store,
        embedding_provider,
        top_k=top_k,
        similarity_threshold=args.threshold,
    )

    report = evaluate_retrieval(
        dataset,
        retriever,
        top_k=top_k,
        dataset_path=args.dataset,
        index_path=args.index,
        index_metadata={
            "embedding_model": store.embedding_model or model_name,
            "source_path": store.source_path,
        },
        similarity_threshold=args.threshold,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_summary(report, output_path)
    return 0


def print_summary(report: dict, output_path: Path) -> None:
    dataset = report["dataset"]
    metrics = report["metrics"]
    print(f"dataset: {dataset['name']}")
    print(f"case_count: {dataset['case_count']}")
    print(
        "top1/top3/top5: "
        f"{metrics['top1_hit']:.3f} / {metrics['top3_hit']:.3f} / {metrics['top5_hit']:.3f}"
    )
    print(
        "recall@1/@3/@5: "
        f"{metrics['recall@1']:.3f} / {metrics['recall@3']:.3f} / {metrics['recall@5']:.3f}"
    )
    print(f"mrr: {metrics['mrr']:.3f}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
