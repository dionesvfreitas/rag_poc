import json
import hashlib
import os
import re
import warnings
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
    spans = source_spans(chunk)
    if spans:
        return ordered_block_ids_from_spans(spans)
    ids = chunk.get("source_block_ids")
    if ids:
        return list(ids)
    source_id = chunk.get("chunk_id")
    return [source_id] if source_id else []


def source_spans(chunk):
    metadata = chunk.get("metadata") or {}
    return [dict(span) for span in chunk.get("source_spans") or metadata.get("source_spans") or []]


def ordered_block_ids_from_spans(spans):
    block_ids = []
    seen = set()
    for span in spans:
        block_id = span.get("block_id")
        if not block_id or block_id in seen:
            continue
        seen.add(block_id)
        block_ids.append(block_id)
    return block_ids


def related_assets(chunk):
    metadata = chunk.get("metadata") or {}
    assets = []
    seen = set()
    for asset in list(chunk.get("related_assets") or []) + list(metadata.get("related_assets") or []):
        if not is_effective_related_asset(asset):
            continue
        key = asset.get("asset_id") or asset.get("asset_uri")
        if key in seen:
            continue
        seen.add(key)
        assets.append(dict(asset))
    return assets


def related_assets_for_block_ids(chunk, block_ids, *, require_target=False):
    block_ids = set(block_ids or [])
    assets = []
    skipped_missing_target = False
    for asset in related_assets(chunk):
        target_block_id = related_asset_target_block_id(asset)
        if target_block_id:
            if require_target and not block_ids:
                continue
            if block_ids and target_block_id not in block_ids:
                continue
        elif require_target:
            skipped_missing_target = True
            continue
        assets.append(asset)
    return assets, skipped_missing_target


def related_asset_target_block_id(asset):
    metadata = asset.get("metadata") or {}
    evidence = asset.get("link_evidence") or {}
    return metadata.get("target_block_id") or evidence.get("target_block_id")


def is_effective_related_asset(asset):
    if not isinstance(asset, dict):
        return False
    strategy = str(asset.get("link_strategy") or "")
    evidence = asset.get("link_evidence") or {}
    decision = evidence.get("decision")
    if strategy.startswith("not_linked") or decision in {"decorative", "not_linked"}:
        return False
    return bool(asset.get("asset_id") and asset.get("asset_uri") and asset.get("asset_type"))


def document_region(chunk):
    metadata = chunk.get("metadata") or {}
    if chunk.get("document_region"):
        return chunk["document_region"]
    if metadata.get("document_region"):
        return metadata["document_region"]
    if list(chunk.get("section_path") or []) == ["front_matter"] or metadata.get("is_front_matter"):
        return "front_matter"
    return None


def source_audit_metadata(source, block_ids=None):
    metadata = source.get("metadata") or {}
    parent_block_ids = source_ids(source)
    source_block_ids = list(block_ids) if block_ids is not None else parent_block_ids
    audit = {
        "source_chunk_id": source.get("chunk_id"),
        "source_block_ids": source_block_ids,
        "source_content_type": source.get("content_type"),
    }
    if block_ids is not None and source_block_ids != parent_block_ids:
        audit["parent_source_block_ids"] = parent_block_ids
    for key, value in metadata.items():
        if key in {"markdown", "related_assets", "source_spans"}:
            continue
        audit.setdefault(key, value)
    region = document_region(source)
    if region:
        audit["document_region"] = region
    if region == "front_matter":
        audit["is_front_matter"] = True
    return audit


def section_key(chunk):
    path = chunk.get("section_path") or []
    if path:
        return tuple(path)
    return ("__unsectioned__", chunk.get("document_id", "document"))


def clause_path(chunk):
    metadata = chunk.get("metadata") or {}
    return list(metadata.get("clause_path") or [])


def context_fields(chunk):
    region = document_region(chunk)
    fields = {
        "document_id": chunk.get("document_id"),
        "section_title": chunk.get("section_title"),
        "subsection_title": chunk.get("subsection_title"),
        "section_path": list(chunk.get("section_path") or []),
        "clause_number": chunk.get("clause_number"),
        "clause_path": clause_path(chunk),
    }
    if region:
        fields["document_region"] = region
    return fields


def provenance_fields(source, content=None, *, split_index=0):
    spans = source_spans(source)
    if content is not None and spans and content != source_text(source):
        ranges = locate_piece_ranges(source_text(source), [content])
        if ranges:
            start, end = ranges[0]
            spans = source_spans_for_text_range(spans, start, end, split_index)
        else:
            spans = []
    block_ids = ordered_block_ids_from_spans(spans) if spans else source_ids(source)
    content_types = list(source.get("content_types") or [])
    if not content_types and source.get("content_type"):
        content_types = [source.get("content_type")]
    assets, skipped_missing_target = related_assets_for_block_ids(source, block_ids)
    fields = {
        "source_chunk_id": source.get("chunk_id"),
        "source_block_ids": block_ids,
        "source_spans": spans,
        "related_assets": assets,
        "content_types": content_types,
    }
    if skipped_missing_target:
        fields["_asset_propagation_skipped_reason"] = "missing_target_block_id"
    return fields


def provenance_fields_for_spans(source, spans, *, require_asset_target=True):
    block_ids = ordered_block_ids_from_spans(spans)
    content_types = list(source.get("content_types") or [])
    if not content_types and source.get("content_type"):
        content_types = [source.get("content_type")]
    assets, skipped_missing_target = related_assets_for_block_ids(
        source, block_ids, require_target=require_asset_target
    )
    fields = {
        "source_chunk_id": source.get("chunk_id"),
        "source_block_ids": block_ids,
        "source_spans": spans,
        "related_assets": assets,
        "content_types": content_types,
    }
    if skipped_missing_target:
        fields["_asset_propagation_skipped_reason"] = "missing_target_block_id"
    return fields


def finalize_provenance_metadata(metadata, provenance):
    reason = provenance.pop("_asset_propagation_skipped_reason", None)
    if reason:
        metadata["asset_propagation_skipped_reason"] = reason


def page_range_fields_for_spans(source, spans):
    pages = []
    for span in spans:
        span_start = span.get("page_start") or span.get("page_no")
        span_end = span.get("page_end") or span.get("page_no")
        if isinstance(span_start, int):
            pages.append(span_start)
        if isinstance(span_end, int):
            pages.append(span_end)
    if pages:
        return {
            "page_start": min(pages),
            "page_end": max(pages),
            "page_total": source.get("page_total"),
            "page_range_strategy": "derived_from_source_spans",
        }
    return {
        "page_start": source.get("page_start") or source.get("page_no"),
        "page_end": source.get("page_end") or source.get("page_no"),
        "page_total": source.get("page_total"),
        "page_range_strategy": "inherited_from_source_chunk",
    }


def apply_page_range_strategy(metadata, page_range):
    metadata["page_range_strategy"] = page_range.pop("page_range_strategy")


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
    provenance = provenance_fields(source)
    metadata = source_audit_metadata(source, provenance["source_block_ids"])
    finalize_provenance_metadata(metadata, provenance)
    page_range = page_range_fields_for_spans(source, provenance["source_spans"])
    apply_page_range_strategy(metadata, page_range)
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
        **page_range,
        **provenance,
        **context_fields(source),
    }


def make_fragment_chunks(parent, child, source, config):
    content = child.get("content") or ""
    ranges = split_semantic_ranges(
        content,
        target_chunk_chars=config.target_chunk_chars,
        max_chunk_chars=config.max_chunk_chars,
    )
    if len(ranges) <= 1:
        return []

    fragments = []
    source_span_list = source_spans(source)
    for index, (piece_start, piece_end) in enumerate(ranges, start=1):
        piece = content[piece_start:piece_end]
        if source_span_list:
            spans = source_spans_for_text_range(source_span_list, piece_start, piece_end, index)
        else:
            spans = []
        provenance = provenance_fields_for_spans(source, spans, require_asset_target=True)
        metadata = source_audit_metadata(source, provenance["source_block_ids"])
        finalize_provenance_metadata(metadata, provenance)
        page_range = page_range_fields_for_spans(source, provenance["source_spans"])
        apply_page_range_strategy(metadata, page_range)
        metadata.update(
            {
                "source_child_chunk_id": child["chunk_id"],
                "source_content_type": source.get("content_type"),
                "fragmented_by": "semantic_text",
            }
        )
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
                "fragment_total": len(ranges),
                "sibling_chunk_ids": [],
                "previous_chunk_id": None,
                "next_chunk_id": None,
                "content": piece,
                "metadata": metadata,
                **page_range,
                **provenance,
                **context_fields(source),
            }
        )
    return fragments


def make_table_chunks(parent, source, config):
    content = source_markdown(source) or source_text(source)
    range_groups = split_markdown_table_row_range_groups(content, config.max_chunk_chars)
    total = len(range_groups)
    chunks = []
    source_span_list = source_spans(source)
    for index, range_group in enumerate(range_groups, start=1):
        table_text = content_from_range_group(content, range_group)
        chunk_id = stable_chunk_id(
            parent["document_id"], "table", f"{source.get('chunk_id')}:{index}:{table_text[:80]}"
        )
        metadata = source.get("metadata") or {}
        if source_span_list:
            spans = source_spans_for_text_ranges(
                source_span_list,
                range_group,
                index if total > 1 else 0,
                content,
            )
        else:
            spans = []
        provenance = provenance_fields_for_spans(source, spans, require_asset_target=total > 1)
        audit_metadata = source_audit_metadata(source, provenance["source_block_ids"])
        finalize_provenance_metadata(audit_metadata, provenance)
        page_range = page_range_fields_for_spans(source, provenance["source_spans"])
        apply_page_range_strategy(audit_metadata, page_range)
        audit_metadata.update(
            {
                "source_content_type": source.get("content_type"),
                "table_syntax_quality": metadata.get("table_syntax_quality"),
                "table_semantic_quality": metadata.get("table_semantic_quality"),
                "table_quality_reasons": list(metadata.get("table_quality_reasons") or []),
            }
        )
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
            "content": table_text,
            "markdown": table_text,
            "table_quality": metadata.get("table_quality"),
            "table_structure": table_structure,
            "metadata": audit_metadata,
            **page_range,
            **provenance,
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
    return [
        text[start:end]
        for start, end in split_semantic_ranges(text, target_chunk_chars, max_chunk_chars)
    ]


def split_semantic_ranges(text, target_chunk_chars=1200, max_chunk_chars=1800):
    if len(text) <= max_chunk_chars:
        return [(0, len(text))]
    units = semantic_units_with_ranges(text, max_chunk_chars)
    ranges = []
    current = []
    current_len = 0

    for start, end in units:
        unit = text[start:end]
        unit_len = len(unit)
        gap_len = start - current[-1][1] if current else 0
        would_exceed = current_len + gap_len + unit_len > max_chunk_chars
        if current and would_exceed:
            ranges.append((current[0][0], current[-1][1]))
            current = []
            current_len = 0
        current.append((start, end))
        current_len += (gap_len if current_len else 0) + unit_len
        if current_len >= target_chunk_chars:
            ranges.append((current[0][0], current[-1][1]))
            current = []
            current_len = 0

    if current:
        ranges.append((current[0][0], current[-1][1]))

    return [(start, end) for start, end in ranges if text[start:end].strip()]


def semantic_units_with_ranges(text, max_chunk_chars):
    units = []
    for paragraph_match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n+|\Z)", text, flags=re.S):
        paragraph_start = paragraph_match.start()
        paragraph_end = paragraph_match.end()
        if paragraph_end - paragraph_start <= max_chunk_chars:
            units.append((paragraph_start, paragraph_match.end()))
            continue

        list_parts = ranges_from_boundary_regex(text, paragraph_start, paragraph_end, LIST_BOUNDARY_RE)
        if len(list_parts) > 1:
            units.extend(split_oversized_ranges(text, list_parts, max_chunk_chars))
            continue

        sentences = ranges_from_split_regex(text, paragraph_start, paragraph_end, SENTENCE_SPLIT_RE)
        if len(sentences) > 1:
            units.extend(split_oversized_ranges(text, sentences, max_chunk_chars))
            continue

        units.extend(split_by_word_boundary_ranges(text, paragraph_start, paragraph_end, max_chunk_chars))
    return units


def semantic_units(text, max_chunk_chars):
    return [text[start:end] for start, end in semantic_units_with_ranges(text, max_chunk_chars)]


def ranges_from_boundary_regex(full_text, start, end, regex):
    start, end = trim_range(full_text, start, end)
    starts = [match.start() for match in regex.finditer(full_text, start, end)]
    if not starts or starts[0] != start:
        starts.insert(0, start)
    ranges = []
    for index, range_start in enumerate(starts):
        range_end = starts[index + 1] if index + 1 < len(starts) else end
        ranges.append(trim_range(full_text, range_start, range_end))
    return [item for item in ranges if item[0] < item[1]]


def ranges_from_split_regex(full_text, start, end, regex):
    start, end = trim_range(full_text, start, end)
    ranges = []
    cursor = start
    for match in regex.finditer(full_text, start, end):
        ranges.append(trim_range(full_text, cursor, match.start()))
        cursor = match.end()
    ranges.append(trim_range(full_text, cursor, end))
    return [item for item in ranges if item[0] < item[1]]


def trim_range(full_text, start, end):
    text_len = len(full_text)
    original = (start, end)
    start = max(0, min(start, text_len))
    end = max(0, min(end, text_len))
    if original != (start, end):
        warnings.warn(
            f"Corrected invalid text range {original} to {(start, end)}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if start > end:
        warnings.warn(
            f"Discarded inverted text range {(start, end)}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return end, end
    while start < end and full_text[start].isspace():
        start += 1
    while end > start and full_text[end - 1].isspace():
        end -= 1
    return start, end


def split_oversized_ranges(text, ranges, max_chunk_chars):
    result = []
    for start, end in ranges:
        if end - start <= max_chunk_chars:
            result.append((start, end))
        else:
            result.extend(split_by_word_boundary_ranges(text, start, end, max_chunk_chars))
    return result


def split_oversized_units(units, max_chunk_chars):
    result = []
    for unit in units:
        if len(unit) <= max_chunk_chars:
            result.append(unit)
        else:
            result.extend(split_by_word_boundary(unit, max_chunk_chars))
    return result


def split_by_word_boundary_ranges(text, start, end, max_chunk_chars):
    ranges = []
    current_start = None
    current_end = None
    for match in re.finditer(r"\S+", text[start:end]):
        word_start = start + match.start()
        word_end = start + match.end()
        if current_start is not None and word_end - current_start > max_chunk_chars:
            ranges.append((current_start, current_end))
            current_start = None
            current_end = None
        if current_start is None:
            current_start = word_start
        current_end = word_end
        if current_end - current_start >= max_chunk_chars:
            ranges.append((current_start, current_end))
            current_start = None
            current_end = None
    if current_start is not None:
        ranges.append((current_start, current_end))
    return ranges


def split_by_word_boundary(text, max_chunk_chars):
    return [text[start:end] for start, end in split_by_word_boundary_ranges(text, 0, len(text), max_chunk_chars)]


def locate_piece_ranges(text, pieces):
    ranges = []
    cursor = 0
    for piece in pieces:
        start = text.find(piece, cursor)
        if start < 0:
            return []
        end = start + len(piece)
        ranges.append((start, end))
        cursor = end
    return ranges


def source_spans_for_text_range(source_span_list, start, end, split_index):
    sliced = []
    for span in source_span_list:
        span_chunk_start = span.get("chunk_start_char")
        span_chunk_end = span.get("chunk_end_char")
        block_start = span.get("block_start_char")
        if not all(isinstance(value, int) for value in (span_chunk_start, span_chunk_end, block_start)):
            continue
        overlap_start = max(start, span_chunk_start)
        overlap_end = min(end, span_chunk_end)
        if overlap_start >= overlap_end:
            continue

        block_start_char = block_start + (overlap_start - span_chunk_start)
        block_end_char = block_start + (overlap_end - span_chunk_start)
        sliced_span = {
            **span,
            "block_start_char": block_start_char,
            "block_end_char": block_end_char,
            "chunk_start_char": overlap_start - start,
            "chunk_end_char": overlap_end - start,
            "separator_before": separator_overlap_for_range(span, start, end),
            "separator_after": None,
            "split_index": split_index,
            "start_char": block_start_char,
            "end_char": block_end_char,
        }
        sliced.append(sliced_span)
    return sliced


def source_spans_for_text_ranges(source_span_list, ranges, split_index, text):
    combined = []
    cursor = 0
    previous_end = None
    for start, end in ranges:
        range_spans = source_spans_for_text_range(source_span_list, start, end, split_index)
        separator = text[previous_end:start] if previous_end is not None and previous_end <= start else None
        separator = separator if separator and separator.strip() == "" else "\n" if previous_end is not None else None
        for index, span in enumerate(range_spans):
            adjusted = {
                **span,
                "chunk_start_char": span["chunk_start_char"] + cursor + len(separator or ""),
                "chunk_end_char": span["chunk_end_char"] + cursor + len(separator or ""),
            }
            if index == 0 and separator:
                adjusted["separator_before"] = f"{separator}{adjusted.get('separator_before') or ''}"
            combined.append(adjusted)
        cursor += len(text[start:end]) + len(separator or "")
        previous_end = end
    return combined


def separator_overlap_for_range(span, start, end):
    separator = span.get("separator_before")
    if not separator:
        return None
    separator_end = span.get("chunk_start_char")
    if not isinstance(separator_end, int):
        return None
    separator_start = separator_end - len(separator)
    overlap_start = max(start, separator_start)
    overlap_end = min(end, separator_end)
    if overlap_start >= overlap_end:
        return None
    return separator[overlap_start - separator_start : overlap_end - separator_start]


def split_markdown_table_rows(markdown, max_chunk_chars):
    return [
        content_from_range_group(markdown, range_group)
        for range_group in split_markdown_table_row_range_groups(markdown, max_chunk_chars)
    ]


def split_markdown_table_row_range_groups(markdown, max_chunk_chars):
    if len(markdown) <= max_chunk_chars:
        return [[(0, len(markdown))]]

    line_ranges = [
        (match.start(), match.end())
        for match in re.finditer(r"[^\n]+", markdown)
        if match.group(0).strip()
    ]
    lines = [markdown[start:end] for start, end in line_ranges]
    if len(lines) <= 3 or not all("|" in line for line in lines[:2]):
        return [[(0, len(markdown))]]

    header_start = line_ranges[0][0]
    header_end = line_ranges[1][1]
    header_range = (header_start, header_end)
    data_rows = line_ranges[2:]
    pieces = []
    current_len = header_end - header_start
    current_rows = []
    for row_start, row_end in data_rows:
        current_end = current_rows[-1][1] if current_rows else header_end
        row_len = row_end - row_start + (row_start - current_end)
        if current_rows and current_len + row_len > max_chunk_chars:
            pieces.append([header_range, (current_rows[0][0], current_rows[-1][1])])
            current_rows = []
            current_len = header_end - header_start
            row_len = row_end - row_start + 1
        current_rows.append((row_start, row_end))
        current_len += row_len
    if current_rows:
        pieces.append([header_range, (current_rows[0][0], current_rows[-1][1])])
    return pieces or [[(0, len(markdown))]]


def content_from_range_group(text, ranges):
    parts = []
    previous_end = None
    for start, end in ranges:
        if previous_end is not None:
            gap = text[previous_end:start]
            parts.append(gap if gap and gap.strip() == "" else "\n")
        parts.append(text[start:end])
        previous_end = end
    return "".join(parts)


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
