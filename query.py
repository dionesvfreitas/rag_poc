import argparse
import textwrap

from mini_rag.embeddings import SentenceTransformersEmbeddingProvider
from mini_rag.retrieval import Retriever
from mini_rag.store import LocalVectorStore
from settings import Settings


def parse_args(argv=None):
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Query a local auditable document index.")
    parser.add_argument("--index", default=settings.index_path, help="Path to document_index.json.")
    parser.add_argument("--query", required=True, help="Question or search text.")
    parser.add_argument("--top-k", type=int, default=settings.top_k, help="Number of chunks to return.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.similarity_threshold,
        help="Minimum cosine similarity score.",
    )
    parser.add_argument("--model", default=None, help="Override embedding model from the index.")
    return parser.parse_args(argv)


def format_results(results):
    if not results:
        return "No chunks matched the query."

    lines = []
    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata") or {}
        chunk = result.get("chunk") or {}
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        page = page_start if page_start == page_end else f"{page_start}-{page_end}"
        section = " > ".join(metadata.get("section_path") or [])
        source_blocks = metadata.get("source_block_ids") or []
        preview = textwrap.shorten(
            " ".join((chunk.get("content") or "").split()),
            width=700,
            placeholder="...",
        )
        lines.extend(
            [
                f"#{index} score={result.get('score'):.4f}",
                f"chunk_id: {chunk.get('chunk_id')}",
                f"source_chunk_id: {metadata.get('source_chunk_id')}",
                f"page: {page}",
                f"section: {section}",
                f"source_block_ids: {', '.join(source_blocks)}",
                "content:",
                preview,
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def main(argv=None):
    args = parse_args(argv)
    store = LocalVectorStore.load(args.index)
    model_name = args.model or store.embedding_model or Settings.from_env().embedding_model
    embedding_provider = SentenceTransformersEmbeddingProvider(model_name)
    retriever = Retriever(
        store,
        embedding_provider,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
    )
    print(format_results(retriever.search(args.query, top_k=args.top_k)))


if __name__ == "__main__":
    main()
