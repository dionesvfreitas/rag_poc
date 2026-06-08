import json
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path


INPUT_JSONL = os.getenv("RAG_INPUT_JSONL", "parsed_sections.jsonl")
OUTPUT_JSONL = os.getenv("RAG_OUTPUT_JSONL", "rag_chunks.jsonl")


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:;])\s+")
LIST_BOUNDARY_RE = re.compile(r"(?m)(?=^\s*(?:[-*\u2022]|\d+(?:\.\d+)*[.)]?)\s+)")


@dataclass(frozen=True)
class RagChunkerConfig:
    target_chunk_chars: int = 1200
    max_chunk_chars: int = 1800
    include_section_context: bool = True

    @classmethod
    def from_env(cls):
        return cls(
            target_chunk_chars=int(os.getenv("RAG_TARGET_CHUNK_CHARS", "1200")),
            max_chunk_chars=int(os.getenv("RAG_MAX_CHUNK_CHARS", "1800")),
            include_section_context=env_bool("RAG_INCLUDE_SECTION_CONTEXT", True),
        )


def env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def write_jsonl(records, output_path):
    with Path(output_path).open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False))
            output_file.write("\n")


def stable_chunk_id(document_id, chunk_type, basis):
    return hashlib.sha256(f"rag:{document_id}:{chunk_type}:{basis}".encode("utf-8")).hexdigest()


def source_text(chunk):
    return chunk.get("text") or chunk.get("page_content") or ""


def source_markdown(chunk):
    metadata = chunk.get("metadata") or {}
    return chunk.get("markdown") or metadata.get("markdown")


def source_ids(chunk):
    ids = chunk.get("source_block_ids")
    if ids:
        return list(ids)
    source_id = chunk.get("chunk_id")
    return [source_id] if source_id else []


def section_key(chunk):
    path = chunk.get("section_path") or []
    if path:
        return tuple(path)
    return ("__unsectioned__", chunk.get("document_id", "document"))


def clause_path(chunk):
    metadata = chunk.get("metadata") or {}
    return list(metadata.get("clause_path") or [])


def context_fields(chunk):
    return {
        "document_id": chunk.get("document_id"),
        "section_title": chunk.get("section_title"),
        "subsection_title": chunk.get("subsection_title"),
        "section_path": list(chunk.get("section_path") or []),
        "clause_number": chunk.get("clause_number"),
        "clause_path": clause_path(chunk),
    }


def page_fields(chunks):
    page_starts = [
        chunk.get("page_start") or chunk.get("page_no")
        for chunk in chunks
        if chunk.get("page_start") or chunk.get("page_no")
    ]
    page_ends = [
        chunk.get("page_end") or chunk.get("page_no")
        for chunk in chunks
        if chunk.get("page_end") or chunk.get("page_no")
    ]
    page_totals = [chunk.get("page_total") for chunk in chunks if chunk.get("page_total")]
    return {
        "page_start": min(page_starts) if page_starts else None,
        "page_end": max(page_ends) if page_ends else None,
        "page_total": max(page_totals) if page_totals else None,
    }


def build_hierarchical_chunks(parsed_chunks, config=None):
    config = config or RagChunkerConfig()
    groups = group_by_section(parsed_chunks)
    records = []
    parent_records = []

    for key, source_chunks in groups.items():
        parent = make_parent_chunk(key, source_chunks)
        parent_records.append(parent)
        section_children = []

        if config.include_section_context:
            section_children.append(make_section_context_chunk(parent, source_chunks))

        for source in source_chunks:
            if source.get("content_type") == "table":
                section_children.extend(make_table_chunks(parent, source, config))
                continue
            if is_redundant_section_title(source, parent):
                continue

            child = make_child_chunk(parent, source)
            section_children.append(child)
            fragments = make_fragment_chunks(parent, child, source, config)
            if fragments:
                child["fragment_chunk_ids"] = [fragment["chunk_id"] for fragment in fragments]
                child["metadata"]["has_fragments"] = True
                section_children.extend(fragments)

        link_section_children(parent, section_children)
        parent["children"] = [child["chunk_id"] for child in section_children]
        records.append(parent)
        records.extend(section_children)

    link_parent_records(parent_records)
    return records


def group_by_section(parsed_chunks):
    groups = {}
    for chunk in parsed_chunks:
        groups.setdefault(section_key(chunk), []).append(chunk)
    return groups


def make_parent_chunk(key, source_chunks):
    first = source_chunks[0]
    document_id = first.get("document_id")
    title = first.get("section_title") or (key[-1] if key else None)
    chunk_id = stable_chunk_id(document_id, "parent", ">".join(str(part) for part in key))
    content = compose_parent_content(title, source_chunks)
    pages = page_fields(source_chunks)
    return {
        "chunk_type": "parent",
        "chunk_id": chunk_id,
        "parent_chunk_id": None,
        "section_root_chunk_id": chunk_id,
        "sibling_chunk_ids": [],
        "previous_chunk_id": None,
        "next_chunk_id": None,
        "section_title": title,
        "subsection_title": None,
        "section_path": list(first.get("section_path") or list(key)),
        "clause_number": None,
        "clause_path": [],
        "document_id": document_id,
        "content": content,
        "children": [],
        **pages,
        "metadata": {
            "source_chunk_ids": [chunk.get("chunk_id") for chunk in source_chunks],
            "source_chunk_count": len(source_chunks),
        },
    }


def compose_parent_content(title, source_chunks):
    parts = []
    if title:
        parts.append(f"Seção: {title}")
    for chunk in source_chunks:
        content = source_markdown(chunk) or source_text(chunk)
        if content and content not in parts:
            parts.append(content)
    return "\n\n".join(parts)


def is_redundant_section_title(source, parent):
    return (
        source.get("content_type") == "title"
        and source.get("clause_number") is not None
        and source_text(source) == parent.get("section_title")
    )


def make_section_context_chunk(parent, source_chunks):
    clauses = []
    subsections = []
    table_count = 0
    for chunk in source_chunks:
        if chunk.get("content_type") == "table":
            table_count += 1
            continue
        clause = chunk.get("clause_number")
        if not clause:
            continue
        subsection_title = chunk.get("subsection_title")
        if subsection_title and subsection_title not in subsections:
            subsections.append(subsection_title)
        if clause not in clauses:
            clauses.append(clause)

    lines = [f"Seção: {parent.get('section_title') or ''}".rstrip()]
    if subsections:
        lines.extend(["", "Subseções:"])
        lines.extend(f"- {title}" for title in subsections)
    if clauses:
        lines.extend(["", "Cláusulas:"])
        lines.extend(f"- {clause}" for clause in clauses)
    if table_count:
        lines.extend(["", f"Tabelas: {table_count}"])

    chunk_id = stable_chunk_id(
        parent["document_id"], "section_context", parent["chunk_id"]
    )
    return {
        "chunk_type": "section_context",
        "chunk_id": chunk_id,
        "parent_chunk_id": parent["chunk_id"],
        "section_root_chunk_id": parent["chunk_id"],
        "sibling_chunk_ids": [],
        "previous_chunk_id": None,
        "next_chunk_id": None,
        "source_chunk_id": None,
        "content": "\n".join(lines),
        "metadata": {"structural_only": True},
        **inherit_parent_context(parent),
    }


def make_child_chunk(parent, source):
    content = source_text(source)
    chunk_id = stable_chunk_id(
        parent["document_id"], "child", source.get("chunk_id") or content[:160]
    )
    metadata = {
        "source_chunk_id": source.get("chunk_id"),
        "source_block_ids": source_ids(source),
        "source_content_type": source.get("content_type"),
    }
    return {
        "chunk_type": "child",
        "chunk_id": chunk_id,
        "parent_chunk_id": parent["chunk_id"],
        "section_root_chunk_id": parent["chunk_id"],
        "sibling_chunk_ids": [],
        "previous_chunk_id": None,
        "next_chunk_id": None,
        "source_chunk_id": source.get("chunk_id"),
        "content": content,
        "metadata": metadata,
        "page_start": source.get("page_start"),
        "page_end": source.get("page_end"),
        "page_total": source.get("page_total"),
        **context_fields(source),
    }


def make_fragment_chunks(parent, child, source, config):
    content = child.get("content") or ""
    pieces = split_semantic_text(
        content,
        target_chunk_chars=config.target_chunk_chars,
        max_chunk_chars=config.max_chunk_chars,
    )
    if len(pieces) <= 1:
        return []

    fragments = []
    for index, piece in enumerate(pieces, start=1):
        chunk_id = stable_chunk_id(
            parent["document_id"], "fragment", f"{child['chunk_id']}:{index}:{piece[:80]}"
        )
        fragments.append(
            {
                "chunk_type": "fragment",
                "chunk_id": chunk_id,
                "parent_chunk_id": parent["chunk_id"],
                "section_root_chunk_id": parent["chunk_id"],
                "source_child_chunk_id": child["chunk_id"],
                "source_chunk_id": source.get("chunk_id"),
                "fragment_index": index,
                "fragment_total": len(pieces),
                "sibling_chunk_ids": [],
                "previous_chunk_id": None,
                "next_chunk_id": None,
                "content": piece,
                "metadata": {
                    "source_content_type": source.get("content_type"),
                    "fragmented_by": "semantic_text",
                },
                "page_start": source.get("page_start"),
                "page_end": source.get("page_end"),
                "page_total": source.get("page_total"),
                **context_fields(source),
            }
        )
    return fragments


def make_table_chunks(parent, source, config):
    content = source_markdown(source) or source_text(source)
    rows = split_markdown_table_rows(content, config.max_chunk_chars)
    total = len(rows)
    chunks = []
    for index, table_text in enumerate(rows, start=1):
        chunk_id = stable_chunk_id(
            parent["document_id"], "table", f"{source.get('chunk_id')}:{index}:{table_text[:80]}"
        )
        metadata = source.get("metadata") or {}
        table_structure = {
            "estimated_rows": metadata.get("table_estimated_rows"),
            "estimated_columns": metadata.get("table_estimated_columns"),
            "pages": metadata.get("table_pages"),
            "continues_previous_page": metadata.get("table_continues_previous_page"),
        }
        record = {
            "chunk_type": "table",
            "chunk_id": chunk_id,
            "parent_chunk_id": parent["chunk_id"],
            "section_root_chunk_id": parent["chunk_id"],
            "sibling_chunk_ids": [],
            "previous_chunk_id": None,
            "next_chunk_id": None,
            "source_chunk_id": source.get("chunk_id"),
            "source_block_ids": source_ids(source),
            "content": table_text,
            "markdown": table_text,
            "table_quality": metadata.get("table_quality"),
            "table_structure": table_structure,
            "metadata": {
                "source_content_type": source.get("content_type"),
                "table_syntax_quality": metadata.get("table_syntax_quality"),
                "table_semantic_quality": metadata.get("table_semantic_quality"),
                "table_quality_reasons": list(metadata.get("table_quality_reasons") or []),
            },
            "page_start": source.get("page_start"),
            "page_end": source.get("page_end"),
            "page_total": source.get("page_total"),
            **context_fields(source),
        }
        if total > 1:
            record["source_table_chunk_id"] = source.get("chunk_id")
            record["table_fragment_index"] = index
            record["table_fragment_total"] = total
            record["metadata"]["fragmented_by"] = "table_rows"
        chunks.append(record)
    return chunks


def split_semantic_text(text, target_chunk_chars=1200, max_chunk_chars=1800):
    if len(text) <= max_chunk_chars:
        return [text]
    units = semantic_units(text, max_chunk_chars)
    pieces = []
    current = []
    current_len = 0

    for unit in units:
        unit_len = len(unit)
        separator_len = 2 if current else 0
        would_exceed = current_len + separator_len + unit_len > max_chunk_chars
        if current and would_exceed:
            pieces.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(unit)
        current_len += (2 if current_len else 0) + unit_len
        if current_len >= target_chunk_chars:
            pieces.append("\n\n".join(current))
            current = []
            current_len = 0

    if current:
        pieces.append("\n\n".join(current))

    return [piece for piece in pieces if piece.strip()]


def semantic_units(text, max_chunk_chars):
    units = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    for paragraph in paragraphs:
        if len(paragraph) <= max_chunk_chars:
            units.append(paragraph)
            continue

        list_parts = [part.strip() for part in LIST_BOUNDARY_RE.split(paragraph) if part.strip()]
        if len(list_parts) > 1:
            units.extend(split_oversized_units(list_parts, max_chunk_chars))
            continue

        sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(paragraph) if part.strip()]
        if len(sentences) > 1:
            units.extend(split_oversized_units(sentences, max_chunk_chars))
            continue

        units.extend(split_by_word_boundary(paragraph, max_chunk_chars))
    return units


def split_oversized_units(units, max_chunk_chars):
    result = []
    for unit in units:
        if len(unit) <= max_chunk_chars:
            result.append(unit)
        else:
            result.extend(split_by_word_boundary(unit, max_chunk_chars))
    return result


def split_by_word_boundary(text, max_chunk_chars):
    words = text.split()
    pieces = []
    current = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > max_chunk_chars:
            pieces.append(" ".join(current))
            current = []
            current_len = 0
        current.append(word)
        current_len += len(word) + (1 if current_len else 0)
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_markdown_table_rows(markdown, max_chunk_chars):
    if len(markdown) <= max_chunk_chars:
        return [markdown]

    lines = [line for line in markdown.splitlines() if line.strip()]
    if len(lines) <= 3 or not all("|" in line for line in lines[:2]):
        return [markdown]

    header = lines[:2]
    data_rows = lines[2:]
    pieces = []
    current = list(header)
    current_len = sum(len(line) + 1 for line in current)
    for row in data_rows:
        row_len = len(row) + 1
        if len(current) > 2 and current_len + row_len > max_chunk_chars:
            pieces.append("\n".join(current))
            current = list(header)
            current_len = sum(len(line) + 1 for line in current)
        current.append(row)
        current_len += row_len
    if len(current) > 2:
        pieces.append("\n".join(current))
    return pieces or [markdown]


def inherit_parent_context(parent):
    return {
        "document_id": parent.get("document_id"),
        "section_title": parent.get("section_title"),
        "subsection_title": None,
        "section_path": list(parent.get("section_path") or []),
        "clause_number": None,
        "clause_path": [],
        "page_start": parent.get("page_start"),
        "page_end": parent.get("page_end"),
        "page_total": parent.get("page_total"),
    }


def link_section_children(parent, children):
    ids = [child["chunk_id"] for child in children]
    for index, child in enumerate(children):
        child["parent_chunk_id"] = parent["chunk_id"]
        child["section_root_chunk_id"] = parent["chunk_id"]
        child["sibling_chunk_ids"] = [
            sibling_id for sibling_id in ids if sibling_id != child["chunk_id"]
        ]
        child["previous_chunk_id"] = ids[index - 1] if index > 0 else None
        child["next_chunk_id"] = ids[index + 1] if index + 1 < len(ids) else None


def link_parent_records(parents):
    ids = [parent["chunk_id"] for parent in parents]
    for index, parent in enumerate(parents):
        parent["sibling_chunk_ids"] = [
            sibling_id for sibling_id in ids if sibling_id != parent["chunk_id"]
        ]
        parent["previous_chunk_id"] = ids[index - 1] if index > 0 else None
        parent["next_chunk_id"] = ids[index + 1] if index + 1 < len(ids) else None


def main():
    config = RagChunkerConfig.from_env()
    parsed_chunks = read_jsonl(INPUT_JSONL)
    rag_chunks = build_hierarchical_chunks(parsed_chunks, config=config)
    write_jsonl(rag_chunks, OUTPUT_JSONL)
    print(
        f"Wrote {len(rag_chunks)} hierarchical RAG chunks to {OUTPUT_JSONL} "
        f"from {INPUT_JSONL}."
    )


if __name__ == "__main__":
    main()
