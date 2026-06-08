import re
from collections import Counter

from parser_core.application.sections import (
    clause_number,
    heading_score,
    is_structural_heading,
    looks_upper_heading,
)
from parser_core.domain.models import ContentType


VISUAL_LABELS = {"picture", "image", "figure", "graphic"}
DATA_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(data:image/[^)]*\)|data:image/[^\s)]+", re.I)
BASE64_RE = re.compile(r"\bbase64\b", re.I)


def normalize_plain_text(text):
    paragraphs = re.split(r"\n\s*\n+", text.strip())
    normalized = []
    for paragraph in paragraphs:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in paragraph.splitlines()]
        normalized.append(" ".join(line for line in lines if line))
    return "\n\n".join(paragraph for paragraph in normalized if paragraph)


def normalized_text(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def sanitize_text_assets(text):
    text = DATA_IMAGE_RE.sub("", text or "")
    text = BASE64_RE.sub("", text)
    return normalize_plain_text(text)


def base64_marker_count(text):
    if not isinstance(text, str) or not text:
        return 0
    data_image_matches = DATA_IMAGE_RE.findall(text)
    remaining = DATA_IMAGE_RE.sub("", text)
    return len(data_image_matches) + len(BASE64_RE.findall(remaining))


def content_type_for(label, text):
    if label in VISUAL_LABELS:
        return ContentType.FIGURE.value
    if label == "table":
        return ContentType.TABLE.value
    if label == "caption":
        return ContentType.CAPTION.value
    if label == "footnote":
        return ContentType.FOOTNOTE.value
    if is_structural_heading(label, text) or looks_upper_heading(text):
        return ContentType.TITLE.value
    if label == "list_item" or clause_number(text):
        return ContentType.LIST_ITEM.value
    return ContentType.PARAGRAPH.value


def table_stats(text):
    lines = [line for line in text.splitlines() if line.strip()]
    table_lines = [line for line in lines if "|" in line]
    column_counts = [len(line.strip("|").split("|")) for line in table_lines]
    reasons = []
    if column_counts and len(set(column_counts)) > 1:
        reasons.append("inconsistent_column_count")
    if any(len(line) > 600 for line in table_lines):
        reasons.append("very_long_rows")
    if not table_lines:
        reasons.append("no_markdown_table_rows")
    quality = "high"
    if len(reasons) == 1:
        quality = "medium"
    elif len(reasons) > 1:
        quality = "low"
    return {
        "table_estimated_rows": len(table_lines) or len(lines),
        "table_estimated_columns": max(column_counts) if column_counts else 0,
        "table_pages": [],
        "table_syntax_quality": quality,
        "table_semantic_quality": "unknown",
        "table_quality": quality,
        "table_quality_reasons": reasons,
    }


def has_edge_position(block):
    if not block.bbox:
        return False
    values = [
        value
        for value in (block.bbox.y0, block.bbox.y1)
        if isinstance(value, int | float)
    ]
    if not values:
        return False
    if all(0 <= value <= 1 for value in values):
        return any(value <= 0.15 or value >= 0.85 for value in values)
    return True


def mark_repeated_headers_footers(
    blocks,
    min_repetition_ratio=0.70,
    max_text_length=120,
):
    if not blocks:
        return []
    page_total = max((block.page_total for block in blocks), default=0)
    if page_total < 3:
        return []

    pages_by_text = {}
    positions_by_text = {}
    original_by_text = {}
    for block in blocks:
        key = normalized_text(block.text)
        if not key or len(key) > max_text_length:
            continue
        pages_by_text.setdefault(key, set()).add(block.page_no)
        positions_by_text.setdefault(key, []).append(has_edge_position(block))
        original_by_text.setdefault(key, block.text)

    warnings = []
    for key, pages in pages_by_text.items():
        if len(pages) < 2:
            continue
        repetition_ratio = len(pages) / page_total
        has_position_signal = any(positions_by_text.get(key, []))
        if repetition_ratio < min_repetition_ratio:
            continue
        warnings.append(f"Repeated header/footer candidate: {original_by_text[key]}")
        for block in blocks:
            if normalized_text(block.text) == key:
                block.metadata["repeated_header_footer_candidate"] = True
                block.metadata["header_footer_repetition_ratio"] = repetition_ratio
                block.metadata["header_footer_position_signal"] = has_position_signal
    return warnings


def normalize_blocks(blocks, config):
    normalized = []
    for block in blocks:
        label = block.metadata.get("label", "")
        is_figure = label in VISUAL_LABELS or block.metadata.get("asset_type") == "image"
        base64_removed = 0
        audited_values = set()
        for value in [block.text, block.metadata.get("markdown")]:
            if isinstance(value, str) and value not in audited_values:
                base64_removed += base64_marker_count(value)
                audited_values.add(value)
        block.text = sanitize_text_assets(block.text)
        if isinstance(block.metadata.get("markdown"), str):
            block.metadata["markdown"] = sanitize_text_assets(block.metadata["markdown"])
        if base64_removed:
            block.metadata["base64_removed"] = (
                block.metadata.get("base64_removed", 0) + base64_removed
            )
        if not block.text and not is_figure:
            continue
        block.content_type = content_type_for(label, block.text)
        block.metadata["heading_confidence"] = heading_score(label, block.text)
        if block.content_type == ContentType.TABLE.value:
            block.metadata.update(table_stats(block.metadata.get("markdown") or block.text))
            if block.page_no is not None:
                block.metadata["table_pages"] = [block.page_no]
        normalized.append(block)
    warnings = mark_repeated_headers_footers(
        normalized,
        min_repetition_ratio=config.header_footer_min_repetition_ratio,
        max_text_length=config.header_footer_max_text_length,
    )
    return normalized, warnings


def content_type_counts(blocks):
    return dict(Counter(block.content_type for block in blocks))
