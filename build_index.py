import argparse
from typing import Callable

from mini_rag.embeddings import EmbeddingProvider, SentenceTransformersEmbeddingProvider
from mini_rag.indexing import build_document_index, default_input_path
from settings import Settings


EmbeddingProviderFactory = Callable[[str], EmbeddingProvider]


def parse_args(argv=None):
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Build a local auditable document index.")
    parser.add_argument("--input", default=str(default_input_path()), help="Input JSONL chunks path.")
    parser.add_argument("--output", default=settings.index_path, help="Output index JSON path.")
    parser.add_argument(
        "--input-type",
        default="auto",
        choices=["auto", "parsed_sections", "rag_chunks"],
        help="Input format. Auto detects rag_chunks by chunk_type.",
    )
    parser.add_argument("--model", default=settings.embedding_model, help="SentenceTransformers model name.")
    return parser.parse_args(argv)


def default_embedding_provider_factory(model_name: str) -> EmbeddingProvider:
    return SentenceTransformersEmbeddingProvider(model_name)


def main(argv=None, embedding_provider_factory: EmbeddingProviderFactory | None = None):
    args = parse_args(argv)
    settings = Settings.from_env()
    settings = Settings(
        embedding_model=args.model,
        top_k=settings.top_k,
        similarity_threshold=settings.similarity_threshold,
        index_path=args.output,
    )
    provider_factory = embedding_provider_factory or default_embedding_provider_factory
    store = build_document_index(
        input_path=args.input,
        output_path=args.output,
        embedding_provider=provider_factory(args.model),
        settings=settings,
        input_type=args.input_type,
    )
    print(
        f"Wrote {len(store.items)} vectors to {args.output} "
        f"from {store.source_path} using {store.embedding_model}."
    )
    return 0


if __name__ == "__main__":
    main()
