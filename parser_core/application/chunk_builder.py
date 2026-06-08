import re
from dataclasses import dataclass

from parser_core.application.sections import clause_number, heading_kind
from parser_core.domain.ids import chunk_id
from parser_core.domain.models import Chunk, ContentType


FINAL_PUNCTUATION_RE = re.compile(r"[.!?:;)\]}|]$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:;])\s+")
LEADING_PUNCTUATION_RE = re.compile(r"^[^\w\s]+")
SENTENCE_BOUNDARY_CHARS = ".!?:;)]}|"


@dataclass(frozen=True)
class ChunkBuilderConfig:
    target_chunk_chars: int = 1200
    max_chunk_chars: int = 2400
    min_chunk_chars: int = 200
    merge_small_chunks: bool = True
    preserve_tables_as_chunks: bool = True


def starts_mid_sentence(text):
    stripped = text.lstrip()
    if not stripped or clause_number(stripped):
        return False
    if LEADING_PUNCTUATION_RE.match(stripped):
        return True
    return stripped[0].islower()


def ends_mid_sentence(text):
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped.startswith("|") and "\n|" in stripped:
        return False
    return FINAL_PUNCTUATION_RE.search(stripped) is None


def starts_mid_sentence_from_range(source_text, start, text):
    if start <= 0:
        return starts_mid_sentence(text)
    if starts_mid_sentence(text):
        return True

    prefix = source_text[:start].rstrip()
    if not prefix:
        return False
    return prefix[-1] not in SENTENCE_BOUNDARY_CHARS


def split_strategy_for_range(source_text, start):
    if start <= 0:
        return "original"
    prefix = source_text[:start]
    if prefix.endswith("\n\n"):
        return "paragraph"
    if prefix.rstrip() and prefix.rstrip()[-1] in SENTENCE_BOUNDARY_CHARS:
        return "sentence"
    return "characters"


def refresh_sentence_metadata(chunk):
    chunk.metadata["starts_mid_sentence"] = starts_mid_sentence(chunk.text)
    chunk.metadata["ends_mid_sentence"] = ends_mid_sentence(chunk.text)
    return chunk


def make_chunk(document_id, chunk_no, blocks, related_assets=None):
    page_numbers = [block.page_no for block in blocks if block.page_no is not None]
    text = "\n\n".join(block.text for block in blocks)
    section_path = list(blocks[0].section_path)
    content_types = sorted({block.content_type for block in blocks})
    metadata = {
        "block_count": len(blocks),
        "labels": sorted({block.metadata.get("label", "") for block in blocks if block.metadata.get("label")}),
        "cross_page": bool(page_numbers and min(page_numbers) != max(page_numbers)),
        "starts_mid_sentence": starts_mid_sentence(text),
        "ends_mid_sentence": ends_mid_sentence(text),
    }
    for block in blocks:
        metadata.update(block.metadata)
    if metadata["cross_page"]:
        metadata["cross_page_reason"] = "page_start_different_from_page_end"
    return Chunk(
        chunk_id=chunk_id(document_id, chunk_no),
        document_id=document_id,
        chunk_no=chunk_no,
        text=text,
        page_start=min(page_numbers) if page_numbers else None,
        page_end=max(page_numbers) if page_numbers else None,
        section_title=section_path[-1] if section_path else None,
        section_path=section_path,
        source_block_ids=[block.block_id for block in blocks],
        source_spans=source_spans_for_blocks(blocks),
        content_types=content_types,
        related_assets=list(related_assets or []),
        metadata=metadata,
    )


def source_spans_for_blocks(blocks, split_index=0):
    return [
        {
            "block_id": block.block_id,
            "start_char": 0,
            "end_char": len(block.text),
            "split_index": split_index,
        }
        for block in blocks
    ]


def split_text_units(text, max_chunk_chars):
    if max_chunk_chars <= 0 or len(text) <= max_chunk_chars:
        return [text]

    pieces = split_units_by_limit(
        [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()],
        max_chunk_chars,
        "\n\n",
    )
    result = []
    for piece in pieces:
        if len(piece) <= max_chunk_chars:
            result.append(piece)
            continue
        result.extend(
            split_units_by_limit(
                [part.strip() for part in SENTENCE_SPLIT_RE.split(piece) if part.strip()],
                max_chunk_chars,
                " ",
            )
        )

    final = []
    for piece in result:
        if len(piece) <= max_chunk_chars:
            final.append(piece)
        else:
            final.extend(split_by_char_limit(piece, max_chunk_chars))
    return [piece for piece in final if piece.strip()]


def split_units_by_limit(units, max_chunk_chars, separator):
    pieces = []
    current = []
    current_len = 0
    separator_len = len(separator)
    for unit in units:
        unit_len = len(unit)
        extra = separator_len if current else 0
        if current and current_len + extra + unit_len > max_chunk_chars:
            pieces.append(separator.join(current))
            current = []
            current_len = 0
        current.append(unit)
        current_len += (separator_len if current_len else 0) + unit_len
    if current:
        pieces.append(separator.join(current))
    return pieces


def split_by_char_limit(text, max_chunk_chars):
    pieces = []
    remaining = text
    while len(remaining) > max_chunk_chars:
        cut = remaining.rfind(" ", 0, max_chunk_chars + 1)
        if cut <= 0:
            cut = max_chunk_chars
        next_start = cut
        while next_start < len(remaining) and remaining[next_start].isspace():
            next_start += 1
        if (
            next_start < len(remaining)
            and LEADING_PUNCTUATION_RE.match(remaining[next_start:])
            and cut < len(remaining)
        ):
            punctuation = LEADING_PUNCTUATION_RE.match(remaining[next_start:]).group(0)
            cut = next_start + len(punctuation)
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def split_large_chunk(chunk, config):
    pieces = split_text_units(chunk.text, config.max_chunk_chars)
    if len(pieces) <= 1:
        return [chunk]

    chunks = []
    piece_ranges = locate_piece_ranges(chunk.text, pieces)
    for index, (piece, (piece_start, piece_end)) in enumerate(
        zip(pieces, piece_ranges, strict=True),
        start=1,
    ):
        split_strategy = split_strategy_for_range(chunk.text, piece_start)
        split = Chunk(
            **{
                **chunk.__dict__,
                "chunk_id": chunk.chunk_id,
                "chunk_no": chunk.chunk_no,
                "text": piece,
                "source_spans": source_spans_for_text_range(
                    chunk.source_spans,
                    piece_start,
                    piece_end,
                    index,
                ),
                "metadata": {
                    **chunk.metadata,
                    "split_index": index,
                    "split_total": len(pieces),
                    "split_strategy": split_strategy,
                    "hard_split": split_strategy == "characters",
                    "starts_mid_sentence": starts_mid_sentence_from_range(
                        chunk.text,
                        piece_start,
                        piece,
                    ),
                    "ends_mid_sentence": ends_mid_sentence(piece),
                    "source_chunk_id": chunk.chunk_id,
                },
            }
        )
        chunks.append(split)
    return chunks


def locate_piece_ranges(text, pieces):
    ranges = []
    cursor = 0
    for piece in pieces:
        start = text.find(piece, cursor)
        if start < 0:
            start = cursor
        end = start + len(piece)
        ranges.append((start, end))
        cursor = end
    return ranges


def source_spans_for_text_range(source_spans, start, end, split_index):
    sliced = []
    span_layout_start = 0
    for index, span in enumerate(source_spans):
        span_length = span["end_char"] - span["start_char"]
        span_layout_end = span_layout_start + span_length
        overlap_start = max(start, span_layout_start)
        overlap_end = min(end, span_layout_end)
        if overlap_start < overlap_end:
            sliced.append(
                {
                    "block_id": span["block_id"],
                    "start_char": span["start_char"] + (overlap_start - span_layout_start),
                    "end_char": span["start_char"] + (overlap_end - span_layout_start),
                    "split_index": split_index,
                }
            )
        span_layout_start = span_layout_end
        if index < len(source_spans) - 1:
            span_layout_start += 2
    return sliced


def renumber_chunks(chunks, document_id):
    for index, chunk in enumerate(chunks, start=1):
        chunk.chunk_no = index
        chunk.chunk_id = chunk_id(document_id, index)
    return chunks


def build_chunks(blocks, document_id, config=None):
    config = config or ChunkBuilderConfig()
    chunks = []
    pending = []
    pending_assets = []

    def flush():
        nonlocal pending, pending_assets
        if pending:
            chunks.append(
                make_chunk(
                    document_id,
                    len(chunks) + 1,
                    pending,
                    related_assets=pending_assets,
                )
            )
            pending = []
            pending_assets = []

    for block in blocks:
        if block.content_type == ContentType.FIGURE.value:
            pending_assets.extend(block.metadata.get("related_assets", []))
            continue

        label = block.metadata.get("label", "")
        if heading_kind(label, block.text):
            flush()
            chunks.append(
                make_chunk(
                    document_id,
                    len(chunks) + 1,
                    [block],
                    related_assets=pending_assets,
                )
            )
            pending_assets = []
            continue

        if block.content_type == ContentType.TABLE.value and config.preserve_tables_as_chunks:
            flush()
            chunks.append(
                make_chunk(
                    document_id,
                    len(chunks) + 1,
                    [block],
                    related_assets=pending_assets,
                )
            )
            pending_assets = []
            continue

        if pending and pending[-1].section_path != block.section_path:
            flush()

        if clause_number(block.text):
            flush()

        if pending and len("\n\n".join(item.text for item in pending + [block])) > config.max_chunk_chars:
            flush()

        pending.append(block)

    flush()

    result = merge_small_chunks(chunks, config) if config.merge_small_chunks else chunks
    split_result = []
    for chunk in result:
        split_result.extend(split_large_chunk(chunk, config))
    return renumber_chunks(split_result, document_id)


def merge_small_chunks(chunks, config):
    merged = []
    for chunk in chunks:
        previous = merged[-1] if merged else None
        can_merge = (
            previous
            and ContentType.TABLE.value not in previous.content_types
            and ContentType.TABLE.value not in chunk.content_types
            and previous.section_path == chunk.section_path
            and len(previous.text) < config.min_chunk_chars
            and len(previous.text) + len(chunk.text) + 2 <= config.max_chunk_chars
        )
        if not can_merge:
            merged.append(chunk)
            continue
        previous.text = "\n\n".join([previous.text, chunk.text])
        previous.page_end = max(
            page for page in [previous.page_end, chunk.page_end] if page is not None
        )
        previous.source_block_ids.extend(chunk.source_block_ids)
        previous.source_spans.extend(chunk.source_spans)
        previous.content_types = sorted(set(previous.content_types) | set(chunk.content_types))
        previous.related_assets.extend(chunk.related_assets)
        previous.metadata["merged_chunk_ids"] = previous.metadata.get("merged_chunk_ids", []) + [chunk.chunk_id]
        refresh_sentence_metadata(previous)
    return merged
