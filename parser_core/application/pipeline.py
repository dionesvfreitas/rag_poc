from dataclasses import dataclass

from parser_core.application.chunk_builder import ChunkBuilderConfig, build_chunks
from parser_core.application.normalizer import normalize_blocks
from parser_core.application.reports import build_report
from parser_core.application.sections import apply_sections


@dataclass(frozen=True)
class ParserPipelineConfig:
    header_footer_min_repetition_ratio: float = 0.70
    header_footer_max_text_length: int = 120
    target_chunk_chars: int = 1200
    max_chunk_chars: int = 2400
    min_chunk_chars: int = 200
    merge_small_chunks: bool = True
    preserve_tables_as_chunks: bool = True


def run_pipeline(parsed_document, config=None, parser_name="docling"):
    config = config or ParserPipelineConfig()
    adapter_warnings = list((parsed_document.metadata or {}).get("warnings") or [])
    blocks, normalizer_warnings = normalize_blocks(parsed_document.blocks, config)
    parsed_document.blocks = apply_sections(blocks)
    chunk_config = ChunkBuilderConfig(
        target_chunk_chars=config.target_chunk_chars,
        max_chunk_chars=config.max_chunk_chars,
        min_chunk_chars=config.min_chunk_chars,
        merge_small_chunks=config.merge_small_chunks,
        preserve_tables_as_chunks=config.preserve_tables_as_chunks,
    )
    chunks = build_chunks(parsed_document.blocks, parsed_document.document_id, chunk_config)
    warnings = adapter_warnings + list(normalizer_warnings)
    report = build_report(parsed_document, chunks, warnings=warnings, parser_name=parser_name)
    return parsed_document, chunks, report
