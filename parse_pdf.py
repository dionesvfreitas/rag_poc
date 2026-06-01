import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", "parsed_sections.jsonl")
ENGINE = os.getenv("DOCLING_ENGINE", "cpu")
NUM_THREADS = int(os.getenv("DOCLING_NUM_THREADS", "8"))


ENGINE_TO_DEVICE = {
    "auto": AcceleratorDevice.AUTO,
    "cpu": AcceleratorDevice.CPU,
    "cuda": AcceleratorDevice.CUDA,
    "mps": AcceleratorDevice.MPS,
    "xpu": AcceleratorDevice.XPU,
}

HEADING_LABELS = {
    "title",
    "section_header",
}

CONTENT_LABELS = {
    "text",
    "list_item",
    "code",
    "formula",
    "caption",
    "footnote",
    "table",
}

CLAUSE_RE = re.compile(
    r"^\s*(?P<number>\d{1,3}(?:\.\d{1,3})*)[.)]?\s+(?P<body>\S.+?)\s*$"
)
ANNEX_RE = re.compile(r"^\s*(ANEXO|ANNEX)\s+[IVXLCDM\d]+(?:[-\s][A-Z0-9]+)?\b", re.I)
NEGATIVE_CLAUSE_PATTERNS = [
    re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"^\s*\d{1,2}h\d{0,2}\b", re.I),
    re.compile(r"^\s*\d+(?:[\s-]\d+){2,}\b"),
    re.compile(r"^\s*\d+\s*%"),
    re.compile(r"^\s*https?://", re.I),
    re.compile(r"^\s*\S+@\S+"),
]
SENTENCE_END_RE = re.compile(r"[.;,]$")
FINAL_PUNCTUATION_RE = re.compile(r"[.!?:;)\]}|]$")


@dataclass(frozen=True)
class ParserConfig:
    engine: str = "cpu"
    num_threads: int = 8
    do_ocr: bool = False
    remove_repeated_headers_footers: bool = True
    header_footer_min_repetition_ratio: float = 0.30
    header_footer_max_text_length: int = 120
    max_chunk_chars: int = 3000
    min_chunk_chars: int = 300
    merge_small_chunks: bool = True
    preserve_tables_as_chunks: bool = True
    debug_artifacts_enabled: bool = False
    debug_artifacts_dir: str = ".artifacts/parser"

    @classmethod
    def from_env(cls):
        return cls(
            engine=os.getenv("DOCLING_ENGINE", "cpu"),
            num_threads=int(os.getenv("DOCLING_NUM_THREADS", "8")),
            do_ocr=env_bool("DOCLING_DO_OCR", False),
            remove_repeated_headers_footers=env_bool(
                "REMOVE_REPEATED_HEADERS_FOOTERS", True
            ),
            header_footer_min_repetition_ratio=float(
                os.getenv("HEADER_FOOTER_MIN_REPETITION_RATIO", "0.30")
            ),
            header_footer_max_text_length=int(
                os.getenv("HEADER_FOOTER_MAX_TEXT_LENGTH", "120")
            ),
            max_chunk_chars=int(os.getenv("MAX_CHUNK_CHARS", "3000")),
            min_chunk_chars=int(os.getenv("MIN_CHUNK_CHARS", "300")),
            merge_small_chunks=env_bool("MERGE_SMALL_CHUNKS", True),
            preserve_tables_as_chunks=env_bool("PRESERVE_TABLES_AS_CHUNKS", True),
            debug_artifacts_enabled=env_bool("DEBUG_ARTIFACTS_ENABLED", False),
            debug_artifacts_dir=os.getenv(
                "DEBUG_ARTIFACTS_DIR", ".artifacts/parser"
            ),
        )


@dataclass
class DocumentBlock:
    text: str
    page_no: int
    page_total: int
    label: str
    level: int = 0
    content_type: str = "paragraph"
    markdown: str | None = None
    bbox: dict | None = None
    metadata: dict = field(default_factory=dict)


def env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def default_input_pdf():
    configured = os.getenv("INPUT_PDF")
    if configured:
        return configured
    pdfs = sorted(Path.cwd().glob("*.pdf"))
    return str(pdfs[0]) if pdfs else "input.pdf"


def normalize_engine(engine):
    normalized = engine.strip().lower()
    if normalized not in ENGINE_TO_DEVICE:
        accepted = ", ".join(sorted(ENGINE_TO_DEVICE))
        raise ValueError(f"Invalid DOCLING_ENGINE={engine!r}. Accepted values: {accepted}.")
    return normalized


def make_converter(engine, num_threads, do_ocr=False):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=num_threads,
        device=ENGINE_TO_DEVICE[engine],
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def item_markdown(item, document):
    if hasattr(item, "export_to_markdown"):
        markdown = item.export_to_markdown(doc=document)
        if markdown:
            return markdown.strip()
    return None


def item_text(item, document):
    if hasattr(item, "text") and item.text:
        return normalize_plain_text(item.text)
    return item_markdown(item, document) or ""


def item_page_no(item):
    provenance = getattr(item, "prov", None)
    if not provenance:
        return None
    return getattr(provenance[0], "page_no", None)


def item_bbox(item):
    provenance = getattr(item, "prov", None)
    if not provenance:
        return None

    bbox = getattr(provenance[0], "bbox", None)
    if bbox is None:
        return None

    data = {}
    for name in ("l", "t", "r", "b", "left", "top", "right", "bottom"):
        if hasattr(bbox, name):
            data[name] = getattr(bbox, name)
    return data or None


def item_label(item):
    label = getattr(item, "label", "")
    return getattr(label, "value", str(label)).lower()


def normalized_text(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def normalize_plain_text(text):
    paragraphs = re.split(r"\n\s*\n+", text.strip())
    normalized = []
    for paragraph in paragraphs:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in paragraph.splitlines()]
        normalized.append(" ".join(line for line in lines if line))
    return "\n\n".join(paragraph for paragraph in normalized if paragraph)


def looks_upper_heading(text):
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 4:
        return False
    upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return upper_ratio >= 0.80 and len(text) <= 180 and not text.rstrip().endswith(".")


def looks_numbered_heading(text):
    return heading_score("text", text) >= 3


def clause_match(text):
    for pattern in NEGATIVE_CLAUSE_PATTERNS:
        if pattern.search(text):
            return None

    match = CLAUSE_RE.match(text)
    if not match:
        return None

    number = match.group("number").rstrip(".")
    body = match.group("body")
    if len(number.split(".")) > 8:
        return None
    if len(number) > 24:
        return None
    if not any(char.isalpha() for char in body):
        return None
    return match


def clause_number(text):
    match = clause_match(text)
    if not match:
        return None
    return match.group("number").rstrip(".")


def clause_level(number):
    return number.count(".")


def clause_tokens(number):
    return [".".join(number.split(".")[: index + 1]) for index in range(len(number.split(".")))]


def clause_body(text):
    match = clause_match(text)
    return match.group("body") if match else ""


def heading_score(label, text):
    if ANNEX_RE.match(text):
        return 5

    score = 0
    number = clause_number(text)
    body = clause_body(text)
    if label in HEADING_LABELS:
        score += 2
    if number and clause_level(number) == 0:
        score += 2
    if number and clause_level(number) <= 1 and len(text) <= 120:
        score += 1
    if looks_upper_heading(text):
        score += 2
    if body and looks_upper_heading(body):
        score += 1
    if text.endswith(":"):
        score += 1
    if SENTENCE_END_RE.search(text.rstrip(":")) and not text.endswith(":"):
        score -= 2
    if number and clause_level(number) >= 2:
        score -= 2
    return score


def heading_kind(label, text):
    if not is_structural_heading(label, text):
        return None

    number = clause_number(text)
    if ANNEX_RE.match(text):
        return "section"
    if not number:
        return "section"
    if clause_level(number) == 0:
        return "section"
    return "subsection"


def section_confidence(label, text):
    return 1.0 if heading_kind(label, text) == "section" else 0.0


def subsection_confidence(label, text):
    if heading_kind(label, text) == "subsection":
        return 1.0
    number = clause_number(text)
    if number and clause_level(number) == 1 and text.endswith(":"):
        return 0.6
    return 0.0


def clause_confidence(label, text, content_type):
    number = clause_number(text)
    if not number:
        return 0.0
    kind = heading_kind(label, text)
    if kind == "section":
        return 0.0
    if kind == "subsection":
        return 0.4
    if content_type in {"list", "paragraph", "mixed"}:
        return 1.0
    return 0.0


def is_structural_heading(label, text):
    if label in HEADING_LABELS and not clause_number(text):
        return True
    return heading_score(label, text) >= 3


def heading_level(text, docling_level=0):
    number = clause_number(text)
    if number:
        return clause_level(number)
    if ANNEX_RE.match(text):
        return 0
    return max(docling_level, 0)


def is_heading_block(block):
    if is_structural_heading(block.label, block.text):
        return True
    if block.content_type == "paragraph" and looks_upper_heading(block.text):
        return True
    return False


def content_type_for(label, text):
    if label == "table":
        return "table"
    if is_structural_heading(label, text) or looks_upper_heading(text):
        return "title"
    if label == "list_item":
        return "list"
    if clause_number(text):
        return "list"
    return "paragraph"


def table_stats(markdown, text):
    raw = markdown or text
    lines = [line for line in raw.splitlines() if line.strip()]
    table_lines = [line for line in lines if "|" in line]
    row_count = len(table_lines) or len(lines)
    column_count = 0
    column_counts = []
    empty_cells = 0
    total_cells = 0
    row_cells = []
    for line in table_lines:
        cells = [cell for cell in line.strip("|").split("|")]
        row_cells.append(cells)
        column_counts.append(len(cells))
        empty_cells += sum(1 for cell in cells if not cell.strip() or set(cell.strip()) <= {"-"})
        total_cells += len(cells)
        column_count = max(column_count, len(cells))
    syntax_reasons = []
    if len(set(column_counts)) > 1:
        syntax_reasons.append("inconsistent_column_count")
    if total_cells and empty_cells / total_cells >= 0.35:
        syntax_reasons.append("many_empty_cells")
    if any(len(line) > 600 for line in table_lines):
        syntax_reasons.append("very_long_rows")
    if not table_lines:
        syntax_reasons.append("no_markdown_table_rows")

    semantic_reasons = []
    data_rows = row_cells[2:] if len(row_cells) > 2 else row_cells
    if data_rows and column_count >= 2:
        first_long_second_short = 0
        lengths = []
        for cells in data_rows:
            normalized_cells = [cell.strip() for cell in cells]
            lengths.extend(len(cell) for cell in normalized_cells if cell)
            if len(normalized_cells) >= 2 and len(normalized_cells[0]) > 80 and len(normalized_cells[1]) < 20:
                first_long_second_short += 1
        if first_long_second_short / max(len(data_rows), 1) >= 0.5:
            semantic_reasons.append("first_column_long_second_column_short")
        if lengths and max(lengths) > 8 * max(min(lengths), 1):
            semantic_reasons.append("high_cell_length_variance")
        if any(any(re.search(r"\b[IVXLCDM]{2,}\b|\b\d+(?:\.\d+)+\b", cell) for cell in cells[1:]) for cells in data_rows):
            semantic_reasons.append("structured_identifier_in_non_first_column")

    syntax_quality = quality_from_reasons(syntax_reasons)
    semantic_quality = "unknown" if not table_lines else quality_from_reasons(semantic_reasons)
    reasons = sorted(set(syntax_reasons + semantic_reasons))
    quality = combine_quality(syntax_quality, semantic_quality)

    return {
        "table_estimated_rows": row_count,
        "table_estimated_columns": column_count,
        "table_syntax_quality": syntax_quality,
        "table_semantic_quality": semantic_quality,
        "table_quality": quality,
        "table_quality_reasons": reasons,
    }


def quality_from_reasons(reasons):
    if not reasons:
        return "high"
    if len(reasons) == 1:
        return "medium"
    return "low"


def combine_quality(*qualities):
    order = {"unknown": 0, "high": 1, "medium": 2, "low": 3}
    reverse_order = {value: key for key, value in order.items()}
    known = [quality for quality in qualities if quality != "unknown"]
    if not known:
        return "unknown"
    return reverse_order[max(order[quality] for quality in known)]


def extract_blocks(document):
    page_total = len(getattr(document, "pages", []))
    blocks = []

    for item, level in document.iterate_items():
        label = item_label(item)
        if label not in HEADING_LABELS and label not in CONTENT_LABELS:
            continue

        text = item_text(item, document)
        page_no = item_page_no(item)
        if not text or page_no is None:
            continue

        markdown = item_markdown(item, document) if label == "table" else None
        content_type = content_type_for(label, text)
        metadata = {
            "heading_confidence": heading_score(label, text),
            "section_confidence": section_confidence(label, text),
            "subsection_confidence": subsection_confidence(label, text),
            "clause_confidence": clause_confidence(label, text, content_type),
        }
        if content_type == "table":
            metadata.update(table_stats(markdown, text))
            metadata["table_pages"] = [page_no]
            metadata["table_continues_previous_page"] = False

        blocks.append(
            DocumentBlock(
                text=text,
                page_no=page_no,
                page_total=page_total,
                label=label,
                level=level,
                content_type=content_type,
                markdown=markdown,
                bbox=item_bbox(item),
                metadata=metadata,
            )
        )

    mark_table_continuations(blocks)
    return blocks


def mark_table_continuations(blocks):
    previous_table = None
    for block in blocks:
        if block.content_type != "table":
            previous_table = None
            continue
        if previous_table and block.page_no == previous_table.page_no + 1:
            block.metadata["table_continues_previous_page"] = True
            block.metadata["table_quality_reasons"] = sorted(
                set(block.metadata.get("table_quality_reasons", []))
                | {"possible_page_continuation"}
            )
            if block.metadata.get("table_quality") == "high":
                block.metadata["table_quality"] = "medium"
            if block.metadata.get("table_semantic_quality") == "high":
                block.metadata["table_semantic_quality"] = "medium"
        previous_table = block


def has_edge_position(block):
    if not block.bbox:
        return False

    top = block.bbox.get("t", block.bbox.get("top"))
    bottom = block.bbox.get("b", block.bbox.get("bottom"))
    if top is None and bottom is None:
        return False

    # Docling bbox coordinates may be absolute or normalized depending on source.
    values = [value for value in (top, bottom) if isinstance(value, int | float)]
    if values and all(0 <= value <= 1 for value in values):
        return any(value <= 0.15 or value >= 0.85 for value in values)
    return True


def remove_repeated_headers_footers(blocks, config):
    if not config.remove_repeated_headers_footers or not blocks:
        return blocks, []

    page_total = max(block.page_total for block in blocks)
    if page_total < 3:
        return blocks, []

    pages_by_text = {}
    positions_by_text = {}
    original_by_text = {}

    for block in blocks:
        key = normalized_text(block.text)
        if not key or len(key) > config.header_footer_max_text_length:
            continue
        pages_by_text.setdefault(key, set()).add(block.page_no)
        positions_by_text.setdefault(key, []).append(has_edge_position(block))
        original_by_text.setdefault(key, block.text)

    candidates = {}
    for key, pages in pages_by_text.items():
        if len(pages) < 2:
            continue
        repetition_ratio = len(pages) / page_total
        has_position_signal = any(positions_by_text.get(key, []))
        threshold = config.header_footer_min_repetition_ratio
        if not has_position_signal:
            threshold = max(threshold, 0.50)
        if repetition_ratio >= threshold:
            candidates[key] = {
                "text": original_by_text[key],
                "pages": sorted(pages),
                "repetition_ratio": repetition_ratio,
                "position_signal": has_position_signal,
            }

    kept = []
    removed = []
    for block in blocks:
        key = normalized_text(block.text)
        candidate = candidates.get(key)
        if not candidate:
            kept.append(block)
            continue

        block.metadata["repeated_header_footer_candidate"] = True
        block.metadata["header_footer_repetition_ratio"] = candidate[
            "repetition_ratio"
        ]
        removed.append(asdict(block))

    return kept, removed


def update_section_path(section_path, title, level):
    if not title:
        return section_path
    if level <= 0:
        return [title]
    if len(section_path) < level:
        return section_path + [title]
    return section_path[:level] + [title]


def chunk_id(document_id, index, text):
    basis = f"{document_id}:{index}:{normalized_text(text)[:160]}"
    return str(uuid5(NAMESPACE_URL, basis))


def parent_chunk_id(document_id, section_path):
    if not section_path:
        return None
    return str(uuid5(NAMESPACE_URL, f"{document_id}:section:{' > '.join(section_path)}"))


def starts_mid_sentence(text):
    stripped = text.lstrip()
    if not stripped:
        return False
    if clause_number(stripped) or ANNEX_RE.match(stripped):
        return False
    return stripped[0].islower()


def ends_mid_sentence(text):
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped.startswith("|") and "\n|" in stripped:
        return False
    return FINAL_PUNCTUATION_RE.search(stripped) is None


def continuation_confidence(text, page_start, page_end):
    confidence = 0.0
    if page_start != page_end:
        confidence += 0.4
    if starts_mid_sentence(text):
        confidence += 0.3
    if ends_mid_sentence(text):
        confidence += 0.3
    return min(confidence, 1.0)


def make_chunk(
    document_id,
    index,
    blocks,
    section_path,
    subsection_title=None,
    section_inherited=False,
):
    page_start = min(block.page_no for block in blocks)
    page_end = max(block.page_no for block in blocks)
    first = blocks[0]
    text = "\n\n".join(block.text for block in blocks)
    content_types = {block.content_type for block in blocks}
    content_type = first.content_type if len(content_types) == 1 else "mixed"
    markdown = first.markdown if len(blocks) == 1 and first.markdown else None
    metadata = {
        "block_count": len(blocks),
        "labels": sorted({block.label for block in blocks}),
    }

    for block in blocks:
        metadata.update(block.metadata)

    if section_inherited:
        metadata["section_title_inherited"] = True

    first_clause_number = clause_number(first.text)
    cross_page = page_start != page_end
    if cross_page:
        metadata["cross_page_reason"] = "page_start_different_from_page_end"
    metadata["cross_page"] = cross_page
    metadata["starts_mid_sentence"] = starts_mid_sentence(text)
    metadata["ends_mid_sentence"] = ends_mid_sentence(text)
    metadata["continuation_confidence"] = continuation_confidence(
        text, page_start, page_end
    )
    return {
        "document_id": document_id,
        "chunk_id": chunk_id(document_id, index, text),
        "parent_chunk_id": parent_chunk_id(document_id, section_path),
        "section_title": section_path[-1] if section_path else None,
        "subsection_title": subsection_title,
        "section_path": list(section_path),
        "clause_number": first_clause_number,
        "page_no": page_start,
        "page_start": page_start,
        "page_end": page_end,
        "page_total": first.page_total,
        "content_type": content_type,
        "page_content": text,
        "markdown": markdown,
        "metadata": metadata,
    }


def split_large_text_chunk(chunk, config):
    text = chunk["page_content"]
    if len(text) <= config.max_chunk_chars:
        return [chunk]

    pieces = []
    paragraphs = re.split(r"\n{2,}", text)
    current = []
    current_len = 0
    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        if current and current_len + paragraph_len + 2 > config.max_chunk_chars:
            pieces.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += paragraph_len + 2
    if current:
        pieces.append("\n\n".join(current))

    parent_id = chunk["parent_chunk_id"] or chunk["chunk_id"]
    split_chunks = []
    for index, piece in enumerate(pieces, start=1):
        split_chunk = dict(chunk)
        split_chunk["chunk_id"] = chunk_id(chunk["document_id"], index, piece)
        split_chunk["parent_chunk_id"] = parent_id
        split_chunk["page_content"] = piece
        split_chunk["metadata"] = dict(chunk["metadata"])
        split_chunk["metadata"]["split_index"] = index
        split_chunk["metadata"]["split_total"] = len(pieces)
        split_chunks.append(split_chunk)
    return split_chunks


def build_chunks(blocks, document_id, config):
    section_path = []
    subsection_title = None
    clause_path = []
    chunks = []
    pending = []
    pending_section_path = []
    pending_subsection_title = None
    section_inherited = False

    def flush_pending():
        nonlocal pending, pending_section_path, pending_subsection_title, section_inherited
        if not pending:
            return
        chunks.append(
            make_chunk(
                document_id,
                len(chunks) + 1,
                pending,
                pending_section_path,
                subsection_title=pending_subsection_title,
                section_inherited=section_inherited,
            )
        )
        pending = []
        pending_subsection_title = None
        section_inherited = bool(pending_section_path)

    for block in blocks:
        block_clause = clause_number(block.text)
        if block_clause:
            clause_path = clause_tokens(block_clause)
            block.metadata["clause_path"] = list(clause_path)
        elif clause_path and block.content_type != "table":
            block.metadata["clause_path"] = list(clause_path)

        kind = heading_kind(block.label, block.text)
        if kind:
            flush_pending()
            if kind == "section":
                level = heading_level(block.text, block.level)
                section_path = update_section_path(section_path, block.text, level)
                subsection_title = None
            else:
                subsection_title = block.text
                block.metadata["subsection_title"] = subsection_title
            chunks.append(
                make_chunk(
                    document_id,
                    len(chunks) + 1,
                    [block],
                    section_path,
                    subsection_title=subsection_title,
                )
            )
            section_inherited = False
            continue

        if block.content_type == "table" and config.preserve_tables_as_chunks:
            flush_pending()
            chunks.append(
                make_chunk(
                    document_id,
                    len(chunks) + 1,
                    [block],
                    section_path,
                    subsection_title=subsection_title,
                )
            )
            section_inherited = bool(section_path)
            continue

        if block_clause:
            flush_pending()

        if (
            pending
            and len("\n\n".join(item.text for item in pending + [block]))
            > config.max_chunk_chars
        ):
            flush_pending()

        pending.append(block)
        pending_section_path = list(section_path)
        pending_subsection_title = subsection_title

    flush_pending()

    if not config.merge_small_chunks:
        return [
            split
            for chunk in chunks
            for split in split_large_text_chunk(chunk, config)
        ]

    merged = merge_small_chunks(chunks, config)
    return [
        split
        for chunk in merged
        for split in split_large_text_chunk(chunk, config)
    ]


def merge_small_chunks(chunks, config):
    if not chunks:
        return chunks

    merged = []
    for chunk in chunks:
        previous = merged[-1] if merged else None
        can_merge = (
            previous
            and previous["content_type"] != "table"
            and chunk["content_type"] != "table"
            and previous["section_path"] == chunk["section_path"]
            and previous["clause_number"] == chunk["clause_number"]
            and len(previous["page_content"]) < config.min_chunk_chars
            and len(previous["page_content"]) + len(chunk["page_content"]) + 2
            <= config.max_chunk_chars
        )
        if not can_merge:
            merged.append(chunk)
            continue

        previous["page_content"] = "\n\n".join(
            [previous["page_content"], chunk["page_content"]]
        )
        previous["page_end"] = max(previous["page_end"], chunk["page_end"])
        previous["metadata"]["block_count"] = previous["metadata"].get(
            "block_count", 1
        ) + chunk["metadata"].get("block_count", 1)
        previous["metadata"]["labels"] = sorted(
            set(previous["metadata"].get("labels", []))
            | set(chunk["metadata"].get("labels", []))
        )
        previous["metadata"]["merged_chunk_ids"] = previous["metadata"].get(
            "merged_chunk_ids", []
        ) + [chunk["chunk_id"]]
        if previous["content_type"] != chunk["content_type"]:
            previous["content_type"] = "mixed"

    return merged


def document_id_for(document):
    name = getattr(document, "name", None) or getattr(document, "origin", None)
    return str(name or "document")


def extract_section_records(document, config=None, document_id=None):
    config = config or ParserConfig.from_env()
    blocks = extract_blocks(document)
    cleaned_blocks, removed_blocks = remove_repeated_headers_footers(blocks, config)
    document_id = document_id or document_id_for(document)
    chunks = build_chunks(cleaned_blocks, document_id, config)
    write_debug_artifacts(config, blocks, removed_blocks, chunks)
    return chunks


def write_debug_artifacts(config, raw_blocks, removed_blocks, chunks):
    if not config.debug_artifacts_enabled:
        return

    artifact_dir = Path(config.debug_artifacts_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl([asdict(block) for block in raw_blocks], artifact_dir / "blocks.raw.jsonl")
    write_jsonl(removed_blocks, artifact_dir / "blocks.removed.jsonl")
    write_jsonl(chunks, artifact_dir / "chunks.jsonl")
    summary = Counter(chunk["content_type"] for chunk in chunks)
    (artifact_dir / "summary.json").write_text(
        json.dumps({"chunk_labels": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(records, output_path):
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False))
            output_file.write("\n")


def main():
    config = ParserConfig.from_env()
    engine = normalize_engine(config.engine)
    input_path = Path(default_input_pdf())
    output_path = Path(OUTPUT_JSONL)

    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    converter = make_converter(
        engine=engine,
        num_threads=config.num_threads,
        do_ocr=config.do_ocr,
    )
    result = converter.convert(input_path)
    records = extract_section_records(result.document, config=config, document_id=input_path.name)
    write_jsonl(records, output_path)

    print(
        f"Wrote {len(records)} semantic chunks to {output_path} "
        f"using engine={engine}, num_threads={config.num_threads}."
    )


if __name__ == "__main__":
    main()
