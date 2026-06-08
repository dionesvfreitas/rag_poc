import re
from dataclasses import dataclass

from parser_core.application.sections import clause_number, heading_kind
from parser_core.domain.ids import chunk_id
from parser_core.domain.models import Chunk, ContentType


FINAL_PUNCTUATION_RE = re.compile(r"[.!?:;)\]}|]$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:;])\s+")
LEADING_PUNCTUATION_RE = re.compile(r"^[^\w\s]+")
SENTENCE_BOUNDARY_CHARS = ".!?:;)]}|"
BLOCK_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class ChunkBuilderConfig:
    target_chunk_chars: int = 1200
    max_chunk_chars: int = 2400
    min_chunk_chars: int = 200
    merge_small_chunks: bool = True
    preserve_tables_as_chunks: bool = True
    max_asset_sequence_distance: int = 3


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
    text = BLOCK_SEPARATOR.join(block.text for block in blocks)
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
    if section_path == ["front_matter"]:
        metadata["is_front_matter"] = True
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


def is_front_matter_record(record):
    return (
        list(getattr(record, "section_path", []) or []) == ["front_matter"]
        or (getattr(record, "metadata", {}) or {}).get("is_front_matter") is True
    )


def is_caption_block(block):
    return (
        block is not None
        and (
            block.content_type == ContentType.CAPTION.value
            or (block.metadata or {}).get("label") == "caption"
        )
    )


def asset_key(asset):
    return asset.get("asset_id") or asset.get("asset_uri")


def bbox_dimension(block, axis):
    bbox = block.bbox
    if bbox is None:
        return None
    start = getattr(bbox, f"{axis}0", None)
    end = getattr(bbox, f"{axis}1", None)
    if start is None or end is None:
        return None
    return abs(end - start)


def bbox_midpoint(block, axis):
    bbox = block.bbox
    if bbox is None:
        return None
    start = getattr(bbox, f"{axis}0", None)
    end = getattr(bbox, f"{axis}1", None)
    if start is None or end is None:
        return None
    return (start + end) / 2


def is_small_figure(block):
    width = bbox_dimension(block, "x")
    height = bbox_dimension(block, "y")
    if width is None or height is None:
        return False
    return width <= 120 and height <= 40


def page_region(block):
    midpoint_y = bbox_midpoint(block, "y")
    if midpoint_y is None:
        return None
    if midpoint_y >= 700:
        return "header"
    if midpoint_y <= 100:
        return "footer"
    return None


def bbox_position_signature(block):
    bbox = block.bbox
    if bbox is None:
        return None
    values = [bbox.x0, bbox.y0, bbox.x1, bbox.y1]
    if any(value is None for value in values):
        return None
    return tuple(round(value / 10) * 10 for value in values)


def next_non_figure_block(blocks, index):
    for next_block in blocks[index + 1 :]:
        if next_block.content_type != ContentType.FIGURE.value:
            return next_block
    return None


def decorative_figure_reasons(blocks):
    signatures = {}
    figures = []
    for index, block in enumerate(blocks):
        if block.content_type != ContentType.FIGURE.value:
            continue
        figures.append((index, block))
        signature = bbox_position_signature(block)
        if signature is None:
            continue
        signatures.setdefault(signature, set()).add(block.page_no)

    reasons = {}
    for index, block in figures:
        if is_caption_block(next_non_figure_block(blocks, index)):
            continue
        key = block.block_id
        if is_front_matter_record(block):
            reasons[key] = "decorative_front_matter_logo"
            continue
        signature = bbox_position_signature(block)
        repeated_position = signature is not None and len(signatures.get(signature, set())) >= 2
        region = page_region(block)
        if repeated_position and region == "header":
            reasons[key] = "decorative_repeated_header"
        elif repeated_position and region == "footer":
            reasons[key] = "decorative_repeated_footer"
        elif repeated_position and is_small_figure(block):
            reasons[key] = "decorative_repeated_position"
        elif is_small_figure(block) and region in {"header", "footer"}:
            reasons[key] = "decorative_small_logo"
    return reasons


def figure_asset_records(block):
    assets = list((block.metadata or {}).get("related_assets") or [])
    if not assets and (block.metadata or {}).get("asset_id"):
        assets = [
            {
                "asset_id": block.metadata.get("asset_id"),
                "asset_uri": block.metadata.get("asset_uri"),
                "asset_type": block.metadata.get("asset_type"),
                "page_no": block.page_no,
                "metadata": dict(block.metadata.get("asset", {}).get("metadata") or {}),
            }
        ]
    return [dict(asset) for asset in assets]


def audited_asset(
    asset,
    source_block,
    *,
    link_strategy,
    link_reason,
    link_score=None,
    target_block=None,
    distance=None,
    decision=None,
    reason_code=None,
):
    metadata = dict(asset.get("metadata") or {})
    decision = decision or ("linked" if target_block is not None else "not_linked")
    reason_code = reason_code or link_strategy
    metadata.update(
        {
            "source_block_id": source_block.block_id,
            "source_sequence_no": source_block.sequence_no,
            "source_page_no": source_block.page_no,
            "source_section_path": list(source_block.section_path),
            "decision": decision,
            "reason": reason_code,
        }
    )
    if target_block is not None:
        metadata.update(
            {
                "target_block_id": target_block.block_id,
                "target_sequence_no": target_block.sequence_no,
                "target_page_no": target_block.page_no,
                "target_section_path": list(target_block.section_path),
            }
        )
    if distance is not None:
        metadata["block_distance"] = distance
    link_evidence = {
        "source_sequence_no": source_block.sequence_no,
        "target_sequence_no": target_block.sequence_no if target_block is not None else None,
        "block_distance": distance,
        "source_page_no": source_block.page_no,
        "target_page_no": target_block.page_no if target_block is not None else None,
        "source_section_path": list(source_block.section_path),
        "target_section_path": list(target_block.section_path) if target_block is not None else None,
        "decision": decision,
        "reason": reason_code,
    }

    return {
        "asset_id": asset.get("asset_id"),
        "asset_type": asset.get("asset_type", "image"),
        "asset_uri": asset.get("asset_uri"),
        "page_no": asset.get("page_no", source_block.page_no),
        "link_strategy": link_strategy,
        "link_reason": link_reason,
        "link_score": link_score,
        "linked_by": "chunk_builder",
        "link_evidence": link_evidence,
        "metadata": metadata,
    }


def set_figure_asset_audit(block, audited_assets, status):
    audited_keys = {asset_key(asset) for asset in audited_assets}
    existing = [
        asset
        for asset in block.metadata.get("related_assets", [])
        if asset.get("link_reason") or asset_key(asset) not in audited_keys
    ]
    block.metadata["related_assets"] = existing + audited_assets
    block.metadata["asset_link_status"] = status


def decorative_assets_for_block(block, reason_code="decorative_front_matter_logo"):
    audited_assets = [
        audited_asset(
            asset,
            block,
            link_strategy="not_linked_decorative",
            link_reason=f"figure was marked decorative: {reason_code}",
            decision="decorative",
            reason_code=reason_code,
        )
        for asset in figure_asset_records(block)
    ]
    set_figure_asset_audit(block, audited_assets, "decorative")
    return audited_assets


def pending_assets_for_block(block):
    return [
        {
            "asset": asset,
            "source_block": block,
        }
        for asset in figure_asset_records(block)
    ]


def audit_pending_assets_for_chunk(pending_assets, blocks, config):
    if not pending_assets:
        return []

    linked_assets = []
    for pending in pending_assets:
        source_block = pending["source_block"]
        asset = pending["asset"]
        target_block, reason_code = nearest_link_target(source_block, blocks, config)
        if target_block is None:
            audited = audited_asset(
                asset,
                source_block,
                link_strategy="not_linked",
                link_reason=f"figure could not be safely linked: {reason_code}",
                decision="not_linked",
                reason_code=reason_code,
            )
            set_figure_asset_audit(source_block, [audited], "unlinked")
            continue

        distance = abs(target_block.sequence_no - source_block.sequence_no)
        if is_caption_block(target_block):
            audited = audited_asset(
                asset,
                source_block,
                target_block=target_block,
                distance=distance,
                link_strategy="caption_detected",
                link_reason="figure has caption immediately after image",
                link_score=1.0,
                decision="linked",
                reason_code="caption_detected",
            )
            linked_assets.append(audited)
            set_figure_asset_audit(source_block, [audited], "linked")
            continue

        audited = audited_asset(
            asset,
            source_block,
            target_block=target_block,
            distance=distance,
            link_strategy="same_page_nearest_text",
            link_reason=(
                f"figure is within {distance} block(s) before nearest text "
                "candidate on same page and same section"
            ),
            link_score=0.75,
            decision="linked",
            reason_code="same_page_nearest_text",
        )
        linked_assets.append(audited)
        set_figure_asset_audit(source_block, [audited], "linked")
    return linked_assets


def mark_unlinked_pending_assets(pending_assets, reason_code):
    for pending in pending_assets:
        source_block = pending["source_block"]
        audited = audited_asset(
            pending["asset"],
            source_block,
            link_strategy="not_linked",
            link_reason=f"figure could not be safely linked: {reason_code}",
            decision="not_linked",
            reason_code=reason_code,
        )
        set_figure_asset_audit(source_block, [audited], "unlinked")


def nearest_link_target(source_block, blocks, config):
    candidates = [
        block
        for block in blocks
        if block.content_type != ContentType.FIGURE.value
        and block.sequence_no > source_block.sequence_no
    ]
    if not candidates:
        return None, "no_posterior_text_candidate"

    same_page_candidates = [
        block
        for block in candidates
        if source_block.page_no is not None and block.page_no == source_block.page_no
    ]
    if not same_page_candidates:
        return None, "different_page"

    same_section_candidates = [
        block
        for block in same_page_candidates
        if list(block.section_path) == list(source_block.section_path)
    ]
    if not same_section_candidates:
        return None, "different_section"

    target_block = min(
        same_section_candidates,
        key=lambda block: (
            abs(block.sequence_no - source_block.sequence_no),
            vertical_distance(source_block, block),
        ),
    )
    distance = abs(target_block.sequence_no - source_block.sequence_no)
    if distance > config.max_asset_sequence_distance:
        return None, "distance_too_large"
    if is_front_matter_record(target_block):
        return None, "front_matter_target"
    return target_block, "same_page_nearest_text"


def vertical_distance(source_block, target_block):
    source_y = bbox_midpoint(source_block, "y")
    target_y = bbox_midpoint(target_block, "y")
    if source_y is None or target_y is None:
        return 0
    return abs(target_y - source_y)


def source_spans_for_blocks(blocks, split_index=0):
    spans = []
    cursor = 0
    for index, block in enumerate(blocks):
        separator_before = BLOCK_SEPARATOR if index else None
        if separator_before:
            cursor += len(separator_before)
        block_end_char = len(block.text)
        chunk_start_char = cursor
        chunk_end_char = cursor + block_end_char
        spans.append(
            {
                "block_id": block.block_id,
                "block_start_char": 0,
                "block_end_char": block_end_char,
                "chunk_start_char": chunk_start_char,
                "chunk_end_char": chunk_end_char,
                "separator_before": separator_before,
                "separator_after": None,
                "split_index": split_index,
                "start_char": 0,
                "end_char": block_end_char,
            }
        )
        cursor = chunk_end_char
    return spans


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
        source_spans = source_spans_for_text_range(
            chunk.source_spans,
            piece_start,
            piece_end,
            index,
        )
        split = Chunk(
            **{
                **chunk.__dict__,
                "chunk_id": chunk.chunk_id,
                "chunk_no": chunk.chunk_no,
                "text": piece,
                "source_block_ids": ordered_block_ids_from_spans(source_spans),
                "source_spans": source_spans,
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


def ordered_block_ids_from_spans(source_spans):
    block_ids = []
    seen = set()
    for span in source_spans:
        block_id = span["block_id"]
        if block_id in seen:
            continue
        seen.add(block_id)
        block_ids.append(block_id)
    return block_ids


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
    for span in source_spans:
        span_chunk_start = span["chunk_start_char"]
        span_chunk_end = span["chunk_end_char"]
        overlap_start = max(start, span_chunk_start)
        overlap_end = min(end, span_chunk_end)
        if overlap_start < overlap_end:
            block_start_char = span["block_start_char"] + (overlap_start - span_chunk_start)
            block_end_char = span["block_start_char"] + (overlap_end - span_chunk_start)
            separator_before = separator_overlap_for_range(span, start, end)
            sliced.append(
                {
                    "block_id": span["block_id"],
                    "block_start_char": block_start_char,
                    "block_end_char": block_end_char,
                    "chunk_start_char": overlap_start - start,
                    "chunk_end_char": overlap_end - start,
                    "separator_before": separator_before,
                    "separator_after": None,
                    "split_index": split_index,
                    "start_char": block_start_char,
                    "end_char": block_end_char,
                }
            )
    return sliced


def separator_overlap_for_range(span, start, end):
    separator = span.get("separator_before")
    if not separator:
        return None
    separator_end = span["chunk_start_char"]
    separator_start = separator_end - len(separator)
    overlap_start = max(start, separator_start)
    overlap_end = min(end, separator_end)
    if overlap_start >= overlap_end:
        return None
    return separator[overlap_start - separator_start : overlap_end - separator_start]


def offset_source_spans(source_spans, offset, separator_before_first=None):
    shifted = []
    position_offset = offset + len(separator_before_first or "")
    for index, span in enumerate(source_spans):
        separator_before = span.get("separator_before")
        if index == 0 and separator_before_first:
            separator_before = f"{separator_before_first}{separator_before or ''}"
        shifted.append(
            {
                **span,
                "chunk_start_char": span["chunk_start_char"] + position_offset,
                "chunk_end_char": span["chunk_end_char"] + position_offset,
                "separator_before": separator_before,
            }
        )
    return shifted


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
    decorative_reasons = decorative_figure_reasons(blocks)

    def flush():
        nonlocal pending, pending_assets
        if pending:
            related_assets = audit_pending_assets_for_chunk(pending_assets, pending, config)
            chunks.append(
                make_chunk(
                    document_id,
                    len(chunks) + 1,
                    pending,
                    related_assets=related_assets,
                )
            )
            pending = []
            pending_assets = []

    for block in blocks:
        if block.content_type == ContentType.FIGURE.value:
            flush()
            if block.block_id in decorative_reasons:
                decorative_assets_for_block(block, decorative_reasons[block.block_id])
            else:
                pending_assets.extend(pending_assets_for_block(block))
            continue

        label = block.metadata.get("label", "")
        if heading_kind(label, block.text):
            flush()
            chunks.append(
                make_chunk(
                    document_id,
                    len(chunks) + 1,
                    [block],
                    related_assets=audit_pending_assets_for_chunk(pending_assets, [block], config),
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
                    related_assets=audit_pending_assets_for_chunk(pending_assets, [block], config),
                )
            )
            pending_assets = []
            continue

        if pending and pending[-1].section_path != block.section_path:
            flush()

        if clause_number(block.text):
            flush()

        if pending and len(BLOCK_SEPARATOR.join(item.text for item in pending + [block])) > config.max_chunk_chars:
            flush()

        pending.append(block)

    flush()
    if pending_assets:
        mark_unlinked_pending_assets(pending_assets, "no_posterior_text_candidate")
        pending_assets = []

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
        previous_text_len = len(previous.text)
        previous.text = BLOCK_SEPARATOR.join([previous.text, chunk.text])
        previous.page_end = max(
            page for page in [previous.page_end, chunk.page_end] if page is not None
        )
        previous.source_block_ids.extend(chunk.source_block_ids)
        previous.source_spans.extend(
            offset_source_spans(
                chunk.source_spans,
                previous_text_len,
                separator_before_first=BLOCK_SEPARATOR,
            )
        )
        previous.content_types = sorted(set(previous.content_types) | set(chunk.content_types))
        previous.related_assets.extend(chunk.related_assets)
        previous.metadata["merged_chunk_ids"] = previous.metadata.get("merged_chunk_ids", []) + [chunk.chunk_id]
        refresh_sentence_metadata(previous)
    return merged
