import json
from pathlib import Path
from typing import Any

from rag_chunker import RagChunkerConfig, build_hierarchical_chunks

from mini_rag.embeddings import EmbeddingProvider, SentenceTransformersEmbeddingProvider
from mini_rag.store import LocalVectorStore
from settings import Settings


INDEXABLE_CHUNK_TYPES = {"child", "fragment", "table"}
SKIPPED_CHUNK_TYPES = {"parent", "section_context"}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def default_input_path():
    rag_chunks = Path("rag_chunks.jsonl")
    if rag_chunks.exists():
        return rag_chunks
    return Path("parsed_sections.jsonl")


def load_chunks(path, input_type="auto"):
    records = read_jsonl(path)
    if input_type == "rag_chunks" or (input_type == "auto" and looks_like_rag_chunks(records)):
        return records
    if input_type in {"parsed_sections", "auto"}:
        return build_hierarchical_chunks(records, config=RagChunkerConfig())
    raise ValueError(f"Unsupported input_type: {input_type}")


def looks_like_rag_chunks(records):
    return bool(records and any("chunk_type" in record for record in records[:5]))


def chunk_content(chunk: dict[str, Any]) -> str:
    return chunk.get("content") or chunk.get("text") or chunk.get("page_content") or ""


def should_index_chunk(chunk: dict[str, Any]) -> bool:
    if not chunk_content(chunk).strip():
        return False
    chunk_type = chunk.get("chunk_type")
    if chunk_type in SKIPPED_CHUNK_TYPES:
        return False
    if chunk_type in INDEXABLE_CHUNK_TYPES:
        return True
    return bool(chunk.get("source_chunk_id") or chunk.get("source_block_ids"))


def audit_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    original_metadata = dict(chunk.get("metadata") or {})
    metadata = dict(original_metadata)
    top_level_fields = [
        "chunk_type",
        "source_chunk_id",
        "source_child_chunk_id",
        "source_table_chunk_id",
        "parent_chunk_id",
        "section_root_chunk_id",
        "page_start",
        "page_end",
        "page_total",
        "section_path",
        "section_title",
        "subsection_title",
        "clause_number",
        "clause_path",
        "source_block_ids",
        "source_spans",
        "related_assets",
        "content_types",
        "previous_chunk_id",
        "next_chunk_id",
        "sibling_chunk_ids",
        "fragment_index",
        "fragment_total",
        "table_fragment_index",
        "table_fragment_total",
        "table_quality",
        "table_structure",
    ]
    for field in top_level_fields:
        if field in chunk:
            metadata[field] = chunk[field]

    metadata.setdefault("source_chunk_id", original_metadata.get("source_chunk_id"))
    metadata.setdefault("page_start", chunk.get("page_no"))
    metadata.setdefault("page_end", chunk.get("page_no"))
    metadata.setdefault("section_path", list(chunk.get("section_path") or []))
    metadata.setdefault("source_block_ids", list(chunk.get("source_block_ids") or []))
    metadata.setdefault("source_spans", list(chunk.get("source_spans") or []))
    metadata.setdefault("related_assets", list(chunk.get("related_assets") or []))
    metadata.setdefault("content_types", list(chunk.get("content_types") or []))
    metadata.setdefault("clause_path", list(original_metadata.get("clause_path") or []))
    return metadata


def build_document_index(
    input_path=None,
    output_path=None,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    settings: Settings | None = None,
    input_type="auto",
):
    settings = settings or Settings.from_env()
    input_path = Path(input_path or default_input_path())
    output_path = Path(output_path or settings.index_path)
    embedding_provider = embedding_provider or SentenceTransformersEmbeddingProvider(settings.embedding_model)

    chunks = [chunk for chunk in load_chunks(input_path, input_type=input_type) if should_index_chunk(chunk)]
    texts = [chunk_content(chunk) for chunk in chunks]
    embeddings = embedding_provider.embed(texts)
    if len(embeddings) != len(chunks):
        raise ValueError("Embedding provider returned a different number of vectors than input texts.")

    store = LocalVectorStore(
        embedding_model=getattr(embedding_provider, "model_name", settings.embedding_model),
        source_path=str(input_path),
    )
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        store.add(
            chunk_id=chunk["chunk_id"],
            content=chunk_content(chunk),
            embedding=embedding,
            metadata=audit_metadata(chunk),
        )
    store.save(output_path)
    return store

