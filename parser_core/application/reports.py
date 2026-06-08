import re

from parser_core.application.normalizer import content_type_counts
from parser_core.domain.models import ContentType, ParserReport


BASE64_AUDIT_RE = re.compile(r"data:image|base64", re.I)
FRONT_MATTER_SECTION_PATH = ["front_matter"]


def asset_key(asset):
    if not isinstance(asset, dict):
        return None
    return asset.get("asset_id") or asset.get("asset_uri")


def block_asset_keys(block):
    keys = set()
    direct_key = block.metadata.get("asset_id") or block.metadata.get("asset_uri")
    if direct_key:
        keys.add(direct_key)
    for asset in block.metadata.get("related_assets", []):
        key = asset_key(asset)
        if key:
            keys.add(key)
    return keys


def document_asset_keys(parsed_document):
    keys = set()
    for asset in (parsed_document.metadata or {}).get("assets", []):
        key = asset_key(asset)
        if key:
            keys.add(key)
    return keys


def chunk_asset_keys(chunks):
    keys = set()
    for chunk in chunks:
        for asset in chunk.related_assets:
            if not is_effective_asset_link(asset):
                continue
            key = asset_key(asset)
            if key:
                keys.add(key)
    return keys


def related_assets_from_records(records):
    assets = []
    for record in records:
        assets.extend(getattr(record, "related_assets", []) or [])
        metadata = getattr(record, "metadata", {}) or {}
        assets.extend(metadata.get("related_assets", []) or [])
    return assets


def asset_links_without_field(assets, field_name):
    return sum(1 for asset in assets if isinstance(asset, dict) and not asset.get(field_name))


def is_effective_asset_link(asset):
    if not isinstance(asset, dict):
        return False
    strategy = asset.get("link_strategy")
    reason = asset.get("link_reason")
    return bool(strategy and reason and not str(strategy).startswith("not_linked"))


def audited_block_asset_keys(blocks, strategy):
    keys = set()
    for block in blocks:
        for asset in (block.metadata or {}).get("related_assets", []):
            if asset.get("link_strategy") != strategy:
                continue
            key = asset_key(asset)
            if key:
                keys.add(key)
    return keys


def base64_removed_count(blocks):
    return sum(
        value
        for block in blocks
        if isinstance((value := block.metadata.get("base64_removed", 0)), int)
    )


def has_base64_residue(record):
    values = [getattr(record, "text", "")]
    metadata = getattr(record, "metadata", {}) or {}
    if isinstance(metadata.get("markdown"), str):
        values.append(metadata["markdown"])
    return any(BASE64_AUDIT_RE.search(value or "") for value in values)


def is_front_matter(record):
    metadata = getattr(record, "metadata", {}) or {}
    return (
        list(getattr(record, "section_path", []) or []) == FRONT_MATTER_SECTION_PATH
        or metadata.get("is_front_matter") is True
    )


def rebuild_chunk_from_spans(blocks_by_id, spans):
    parts = []
    for span in sorted(spans, key=lambda item: item.get("chunk_start_char", 0)):
        if span.get("separator_before"):
            parts.append(span["separator_before"])
        block = blocks_by_id.get(span.get("block_id"))
        if block is None:
            return None
        start = span.get("block_start_char")
        end = span.get("block_end_char")
        if not isinstance(start, int) or not isinstance(end, int):
            return None
        if start < 0 or end < start or end > len(block.text):
            return None
        parts.append(block.text[start:end])
        if span.get("separator_after"):
            parts.append(span["separator_after"])
    return "".join(parts)


def chunk_rebuildable_from_spans(chunk, blocks_by_id):
    spans = chunk.source_spans or []
    if not spans:
        return False
    rebuilt = rebuild_chunk_from_spans(blocks_by_id, spans)
    return rebuilt == chunk.text


def add_warning(warnings, message):
    if message not in warnings:
        warnings.append(message)


def audit_warnings(
    warnings,
    *,
    blocks_without_page,
    blocks_without_section,
    chunks_without_section,
    figures_detected,
    figures_saved,
    figures_linked_to_chunks,
    figures_unlinked,
    figures_marked_decorative,
    figure_blocks_without_assets,
    base64_removed,
    base64_residual_blocks,
    base64_residual_chunks,
    chunks_split,
    chunks_mid_sentence,
    chunks_hard_split,
    chunks_without_source_spans,
    chunks_not_rebuildable_from_spans,
    asset_links_without_strategy,
    asset_links_without_reason,
):
    if base64_removed:
        add_warning(warnings, f"Removed {base64_removed} base64/data-image marker(s).")
    if base64_residual_blocks or base64_residual_chunks:
        add_warning(
            warnings,
            (
                "Residual base64/data-image markers found after normalization "
                f"({base64_residual_blocks} block(s), {base64_residual_chunks} chunk(s))."
            ),
        )
    if figures_detected and figure_blocks_without_assets:
        add_warning(
            warnings,
            f"{figure_blocks_without_assets} figure block(s) have no saved asset metadata.",
        )
    if figures_unlinked:
        add_warning(warnings, f"{figures_unlinked} figure asset(s) could not be safely linked.")
    if figures_marked_decorative:
        add_warning(warnings, f"{figures_marked_decorative} figure asset(s) marked decorative.")
    if chunks_without_section:
        add_warning(warnings, f"{chunks_without_section} chunk(s) without section_path.")
    if chunks_without_source_spans:
        add_warning(warnings, f"{chunks_without_source_spans} chunk(s) without source_spans.")
    if chunks_not_rebuildable_from_spans:
        add_warning(
            warnings,
            f"{chunks_not_rebuildable_from_spans} chunk(s) not rebuildable from source_spans.",
        )
    if asset_links_without_strategy:
        add_warning(warnings, f"{asset_links_without_strategy} asset link(s) without link_strategy.")
    if asset_links_without_reason:
        add_warning(warnings, f"{asset_links_without_reason} asset link(s) without link_reason.")
    if blocks_without_page:
        add_warning(warnings, f"{blocks_without_page} block(s) without page_no.")
    if blocks_without_section:
        add_warning(warnings, f"{blocks_without_section} block(s) without section_path.")
    if chunks_split:
        add_warning(warnings, f"{chunks_split} chunk(s) generated by split operation.")
    if chunks_mid_sentence:
        add_warning(warnings, f"{chunks_mid_sentence} chunk(s) start mid-sentence.")
    if chunks_hard_split:
        add_warning(warnings, f"{chunks_hard_split} chunk(s) required hard character split.")
    return warnings


def build_report(parsed_document, chunks, warnings=None, parser_name="docling"):
    warnings = list(warnings or [])
    blocks = parsed_document.blocks
    figure_blocks = [block for block in blocks if block.content_type == ContentType.FIGURE.value]
    blocks_by_id = {block.block_id: block for block in blocks}
    saved_asset_keys = document_asset_keys(parsed_document)
    linked_asset_keys = chunk_asset_keys(chunks)
    decorative_asset_keys = audited_block_asset_keys(blocks, "not_linked_decorative")
    explicitly_unlinked_asset_keys = audited_block_asset_keys(blocks, "not_linked")
    figures_unlinked = len(explicitly_unlinked_asset_keys | (saved_asset_keys - linked_asset_keys - decorative_asset_keys))
    pages_with_text = {block.page_no for block in blocks if block.page_no is not None and block.text}
    all_pages = set(range(1, parsed_document.page_total + 1))
    block_lengths = [len(block.text) for block in blocks]
    chunk_lengths = [len(chunk.text) for chunk in chunks]
    blocks_without_page = sum(1 for block in blocks if block.page_no is None)
    front_matter_blocks = sum(1 for block in blocks if is_front_matter(block))
    front_matter_chunks = sum(1 for chunk in chunks if is_front_matter(chunk))
    blocks_without_section = sum(
        1 for block in blocks if not block.section_path and not is_front_matter(block)
    )
    chunks_without_section = sum(
        1 for chunk in chunks if not chunk.section_path and not is_front_matter(chunk)
    )
    chunks_split = sum(1 for chunk in chunks if (chunk.metadata or {}).get("split_total", 1) > 1)
    chunks_mid_sentence = sum(
        1 for chunk in chunks if (chunk.metadata or {}).get("starts_mid_sentence")
    )
    chunks_hard_split = sum(1 for chunk in chunks if (chunk.metadata or {}).get("hard_split"))
    source_spans_total = sum(len(chunk.source_spans or []) for chunk in chunks)
    chunks_with_source_spans = sum(1 for chunk in chunks if chunk.source_spans)
    chunks_without_source_spans = len(chunks) - chunks_with_source_spans
    chunks_rebuildable_from_spans = sum(
        1 for chunk in chunks if chunk_rebuildable_from_spans(chunk, blocks_by_id)
    )
    chunks_not_rebuildable_from_spans = chunks_with_source_spans - chunks_rebuildable_from_spans
    all_related_assets = related_assets_from_records([*blocks, *chunks])
    asset_links_without_strategy = asset_links_without_field(all_related_assets, "link_strategy")
    asset_links_without_reason = asset_links_without_field(all_related_assets, "link_reason")
    base64_removed = base64_removed_count(blocks)
    warnings = audit_warnings(
        warnings,
        blocks_without_page=blocks_without_page,
        blocks_without_section=blocks_without_section,
        chunks_without_section=chunks_without_section,
        figures_detected=len(figure_blocks),
        figures_saved=len(saved_asset_keys),
        figures_linked_to_chunks=len(linked_asset_keys),
        figures_unlinked=figures_unlinked,
        figures_marked_decorative=len(decorative_asset_keys),
        figure_blocks_without_assets=sum(1 for block in figure_blocks if not block_asset_keys(block)),
        base64_removed=base64_removed,
        base64_residual_blocks=sum(1 for block in blocks if has_base64_residue(block)),
        base64_residual_chunks=sum(1 for chunk in chunks if has_base64_residue(chunk)),
        chunks_split=chunks_split,
        chunks_mid_sentence=chunks_mid_sentence,
        chunks_hard_split=chunks_hard_split,
        chunks_without_source_spans=chunks_without_source_spans,
        chunks_not_rebuildable_from_spans=chunks_not_rebuildable_from_spans,
        asset_links_without_strategy=asset_links_without_strategy,
        asset_links_without_reason=asset_links_without_reason,
    )
    return ParserReport(
        document_id=parsed_document.document_id,
        parser_name=parser_name,
        source_name=parsed_document.source_name,
        page_total=parsed_document.page_total,
        total_blocks=len(blocks),
        total_chunks=len(chunks),
        content_types_count=content_type_counts(blocks),
        pages_detected=sorted(pages_with_text),
        pages_without_text=sorted(all_pages - pages_with_text),
        blocks_without_page=blocks_without_page,
        blocks_without_section=blocks_without_section,
        avg_block_chars=sum(block_lengths) / len(block_lengths) if block_lengths else 0.0,
        avg_chunk_chars=sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0.0,
        max_chunk_chars=max(chunk_lengths) if chunk_lengths else 0,
        figures_detected=len(figure_blocks),
        figures_saved=len(saved_asset_keys),
        figures_linked_to_chunks=len(linked_asset_keys),
        figures_unlinked=figures_unlinked,
        figures_marked_decorative=len(decorative_asset_keys),
        asset_links_without_strategy=asset_links_without_strategy,
        asset_links_without_reason=asset_links_without_reason,
        source_spans_total=source_spans_total,
        chunks_with_source_spans=chunks_with_source_spans,
        chunks_without_source_spans=chunks_without_source_spans,
        chunks_rebuildable_from_spans=chunks_rebuildable_from_spans,
        chunks_not_rebuildable_from_spans=chunks_not_rebuildable_from_spans,
        base64_removed=base64_removed,
        front_matter_blocks=front_matter_blocks,
        front_matter_chunks=front_matter_chunks,
        chunks_without_section=chunks_without_section,
        chunks_split=chunks_split,
        chunks_mid_sentence=chunks_mid_sentence,
        chunks_hard_split=chunks_hard_split,
        warnings=warnings,
    )
