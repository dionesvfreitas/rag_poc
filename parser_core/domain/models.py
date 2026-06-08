from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ContentType(StrEnum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_BREAK = "page_break"
    UNKNOWN = "unknown"


class AssetType(StrEnum):
    IMAGE = "image"
    TABLE = "table"
    FIGURE = "figure"
    CHART = "chart"
    DIAGRAM = "diagram"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None
    coord_origin: str | None = None


@dataclass
class DocumentAsset:
    asset_id: str
    asset_uri: str
    asset_type: AssetType | str
    page_no: int | None = None
    source_block_id: str | None = None
    bbox: BoundingBox | None = None
    caption: str | None = None
    ocr_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.asset_type, AssetType):
            try:
                self.asset_type = AssetType(self.asset_type)
            except ValueError:
                self.asset_type = AssetType.UNKNOWN


@dataclass
class ParsedBlock:
    block_id: str
    document_id: str
    page_no: int | None
    page_total: int
    sequence_no: int
    content_type: str
    text: str
    section_title: str | None = None
    section_path: list[str] = field(default_factory=list)
    bbox: BoundingBox | None = None
    parser_name: str = "unknown"
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    document_id: str
    source_path: str
    source_name: str
    page_total: int
    blocks: list[ParsedBlock] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    chunk_no: int
    text: str
    page_start: int | None
    page_end: int | None
    section_title: str | None = None
    section_path: list[str] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    source_spans: list[dict[str, Any]] = field(default_factory=list)
    content_types: list[str] = field(default_factory=list)
    related_assets: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParserReport:
    document_id: str
    parser_name: str
    source_name: str
    page_total: int
    total_blocks: int
    total_chunks: int
    content_types_count: dict[str, int]
    pages_detected: list[int]
    pages_without_text: list[int]
    blocks_without_page: int
    blocks_without_section: int
    avg_block_chars: float
    avg_chunk_chars: float
    max_chunk_chars: int
    figures_detected: int = 0
    figures_saved: int = 0
    figures_linked_to_chunks: int = 0
    figures_unlinked: int = 0
    figures_marked_decorative: int = 0
    asset_links_without_strategy: int = 0
    asset_links_without_reason: int = 0
    source_spans_total: int = 0
    chunks_with_source_spans: int = 0
    chunks_without_source_spans: int = 0
    chunks_rebuildable_from_spans: int = 0
    chunks_not_rebuildable_from_spans: int = 0
    base64_removed: int = 0
    front_matter_blocks: int = 0
    front_matter_chunks: int = 0
    chunks_without_section: int = 0
    chunks_split: int = 0
    chunks_mid_sentence: int = 0
    chunks_hard_split: int = 0
    warnings: list[str] = field(default_factory=list)


def to_dict(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
