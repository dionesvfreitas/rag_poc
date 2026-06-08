import json
from dataclasses import asdict
from pathlib import Path


def record_to_dict(record):
    if isinstance(record, dict):
        return record
    data = asdict(record)
    if data.get("bbox") is None:
        return data
    return data


def write_jsonl(records, output_path):
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record_to_dict(record), ensure_ascii=False))
            output_file.write("\n")


def write_json(record, output_path):
    Path(output_path).write_text(
        json.dumps(record_to_dict(record), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_markdown(blocks, output_path):
    lines = []
    for block in blocks:
        section = " > ".join(block.section_path)
        lines.append(f"<!-- page_no={block.page_no} block_id={block.block_id} section_path={section} -->")
        lines.append(block.text)
        lines.append("")
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
