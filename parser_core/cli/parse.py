import os
from dataclasses import asdict
from pathlib import Path

from parser_core.application.pipeline import ParserPipelineConfig, run_pipeline
from parser_core.infrastructure.docling_adapter import DoclingPdfParser
from parser_core.infrastructure.exporters import write_json, write_jsonl, write_markdown
from parser_core.registry import ParserRegistry


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


def config_from_env():
    return ParserPipelineConfig(
        header_footer_min_repetition_ratio=float(
            os.getenv("HEADER_FOOTER_MIN_REPETITION_RATIO", "0.70")
        ),
        header_footer_max_text_length=int(
            os.getenv("HEADER_FOOTER_MAX_TEXT_LENGTH", "120")
        ),
        target_chunk_chars=int(os.getenv("TARGET_CHUNK_CHARS", "1200")),
        max_chunk_chars=int(os.getenv("MAX_CHUNK_CHARS", "2400")),
        min_chunk_chars=int(os.getenv("MIN_CHUNK_CHARS", "200")),
        merge_small_chunks=env_bool("MERGE_SMALL_CHUNKS", True),
        preserve_tables_as_chunks=env_bool("PRESERVE_TABLES_AS_CHUNKS", True),
    )


def legacy_chunk_record(chunk):
    data = asdict(chunk)
    data["page_content"] = data["text"]
    data["content_type"] = (
        data["content_types"][0] if len(data["content_types"]) == 1 else "mixed"
    )
    data["page_no"] = data["page_start"]
    data["markdown"] = data["metadata"].get("markdown")
    data["clause_number"] = data["metadata"].get("clause_number")
    data["subsection_title"] = data["metadata"].get("subsection_title")
    return data


def main():
    input_path = Path(default_input_pdf())
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    output_jsonl = Path(os.getenv("OUTPUT_JSONL", "parsed_sections.jsonl"))
    blocks_jsonl = Path(os.getenv("BLOCKS_OUTPUT_JSONL", "normalized_blocks.jsonl"))
    markdown_output = Path(os.getenv("MARKDOWN_OUTPUT", "normalized.md"))
    report_output = Path(os.getenv("REPORT_OUTPUT", "parser_report.json"))
    assets_output = Path(os.getenv("ASSETS_OUTPUT_JSONL", "assets.jsonl"))

    registry = ParserRegistry()
    registry.register("docling", DoclingPdfParser.from_env())
    parser_name = os.getenv("PARSER_NAME", "docling")
    parser = registry.get(parser_name)

    parsed_document = parser.parse(input_path)
    parsed_document, chunks, report = run_pipeline(
        parsed_document,
        config=config_from_env(),
        parser_name=parser_name,
    )

    write_jsonl(parsed_document.blocks, blocks_jsonl)
    write_jsonl([legacy_chunk_record(chunk) for chunk in chunks], output_jsonl)
    write_jsonl(parsed_document.metadata.get("assets", []), assets_output)
    write_markdown(parsed_document.blocks, markdown_output)
    write_json(report, report_output)
    print(
        f"Wrote {len(chunks)} chunks to {output_jsonl}, "
        f"{len(parsed_document.blocks)} blocks to {blocks_jsonl}, "
        f"and report to {report_output}."
    )


if __name__ == "__main__":
    main()
