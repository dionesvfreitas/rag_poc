import importlib.metadata
import json
import os
from pathlib import Path

from parser_core.application.assets import document_asset_from_figure_block
from parser_core.application.normalizer import normalize_plain_text
from parser_core.domain.ids import block_id, document_id_for_path
from parser_core.domain.models import BoundingBox, ParsedBlock, ParsedDocument, to_dict
from parser_core.infrastructure.asset_store import LocalAssetStore, asset_record, decode_data_uri


ENGINE_TO_DEVICE = {"auto", "cpu", "cuda", "mps", "xpu"}
VISUAL_LABELS = {"picture", "image", "figure", "graphic"}
CONTENT_LABELS = {
    "text",
    "list_item",
    "code",
    "formula",
    "caption",
    "footnote",
    "table",
    "picture",
    "image",
    "figure",
    "graphic",
    "section_header",
    "title",
}


class DoclingPdfParser:
    name = "docling"

    def __init__(self, engine="cpu", num_threads=8, do_ocr=False, asset_store=None):
        self.engine = normalize_engine(engine)
        self.num_threads = num_threads
        self.do_ocr = do_ocr
        self.asset_store = asset_store

    @classmethod
    def from_env(cls):
        return cls(
            engine=os.getenv("DOCLING_ENGINE", "cpu"),
            num_threads=int(os.getenv("DOCLING_NUM_THREADS", "8")),
            do_ocr=env_bool("DOCLING_DO_OCR", False),
            asset_store=LocalAssetStore(os.getenv("OUTPUT_IMAGES_DIR", "outputs/images")),
        )

    def parse(self, source_path):
        source = Path(source_path)
        converter = make_converter(self.engine, self.num_threads, self.do_ocr)
        result = converter.convert(source)
        return docling_document_to_parsed_document(
            result.document,
            source,
            asset_store=self.asset_store,
        )


def env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_engine(engine):
    normalized = engine.strip().lower()
    if normalized not in ENGINE_TO_DEVICE:
        accepted = ", ".join(sorted(ENGINE_TO_DEVICE))
        raise ValueError(f"Invalid DOCLING_ENGINE={engine!r}. Accepted values: {accepted}.")
    return normalized


def installed_docling_version():
    try:
        return importlib.metadata.version("docling")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def make_converter(engine, num_threads, do_ocr=False):
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    engine_to_device = {
        "auto": AcceleratorDevice.AUTO,
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,
        "xpu": AcceleratorDevice.XPU,
    }
    pipeline_options = PdfPipelineOptions(
        generate_picture_images=True,
    )
    pipeline_options.do_ocr = do_ocr
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=num_threads,
        device=engine_to_device[engine],
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def item_label(item):
    label = getattr(item, "label", "")
    return getattr(label, "value", str(label)).lower()


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


def visual_item_image(item, document, markdown):
    for method_name in ("get_image", "get_pil_image"):
        method = getattr(item, method_name, None)
        if not callable(method):
            continue
        for args in ((document,), ()):
            try:
                image = method(*args)
            except TypeError:
                continue
            if image is not None:
                return image
    for attr_name in ("image", "pil_image", "_image"):
        image = getattr(item, attr_name, None)
        if image is not None:
            return image
    data, _ = decode_data_uri(markdown)
    return markdown if data is not None else None


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
    x0 = first_attr(bbox, ("l", "left", "x0"))
    y0 = first_attr(bbox, ("t", "top", "y0"))
    x1 = first_attr(bbox, ("r", "right", "x1"))
    y1 = first_attr(bbox, ("b", "bottom", "y1"))
    coord_origin = first_attr(bbox, ("coord_origin",))
    if all(value is None for value in (x0, y0, x1, y1)):
        return None
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1, coord_origin=coord_origin)


def first_attr(obj, names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def make_block(
    document_id,
    page_total,
    sequence_no,
    text,
    page_no=None,
    label="text",
    level=0,
    markdown=None,
    bbox=None,
    metadata=None,
):
    metadata = metadata or {}
    return ParsedBlock(
        block_id=block_id(document_id, page_no, sequence_no),
        document_id=document_id,
        page_no=page_no,
        page_total=page_total,
        sequence_no=sequence_no,
        content_type="unknown",
        text=normalize_plain_text(text),
        bbox=bbox,
        parser_name="docling",
        metadata={
            "label": label,
            "level": level,
            "markdown": markdown,
            **metadata,
        },
    )


def make_asset_metadata(asset):
    if asset is None:
        return {}
    data = asset_record(asset)
    return {
        "asset_id": data["asset_id"],
        "asset_uri": data["asset_uri"],
        "asset_type": data["asset_type"],
        "asset": data,
        "related_assets": [data],
    }


def extract_native_blocks(document, document_id, page_total, warnings, asset_store=None):
    blocks = []
    if not hasattr(document, "iterate_items"):
        warnings.append("Docling native extraction skipped: document has no iterate_items.")
        return blocks

    for item, level in document.iterate_items():
        label = item_label(item)
        if label not in CONTENT_LABELS:
            continue
        page_no = item_page_no(item)
        markdown = item_markdown(item, document)
        metadata = {}
        if label in VISUAL_LABELS:
            asset = None
            if asset_store is not None:
                image = visual_item_image(item, document, markdown)
                asset = asset_store.save_figure(
                    image,
                    page_no,
                    len([block for block in blocks if block.metadata.get("asset_type") == "image"]) + 1,
                    metadata={"source_label": label},
                )
                if asset is None:
                    warnings.append(f"Docling image asset could not be saved for page {page_no}.")
            metadata.update(make_asset_metadata(asset))
            text = ""
        else:
            text = item_text(item, document)
        if not text and label not in VISUAL_LABELS:
            continue
        markdown = markdown if label == "table" else None
        blocks.append(
            make_block(
                document_id=document_id,
                page_total=page_total,
                sequence_no=len(blocks) + 1,
                text=text,
                page_no=page_no,
                label=label,
                level=level,
                markdown=markdown,
                bbox=item_bbox(item),
                metadata=metadata,
            )
        )
    return blocks


def public_export_value(document, method_names, warnings):
    for method_name in method_names:
        if not hasattr(document, method_name):
            continue
        method = getattr(document, method_name)
        if not callable(method):
            continue
        try:
            return method_name, method()
        except TypeError as exc:
            warnings.append(f"Docling export {method_name} failed: {exc}.")
        except Exception as exc:
            warnings.append(f"Docling export {method_name} failed: {exc}.")
    return None, None


def export_dict(document, warnings):
    method_name, value = public_export_value(
        document,
        ("export_to_dict", "to_dict", "model_dump"),
        warnings,
    )
    if isinstance(value, dict):
        return method_name, value

    method_name, value = public_export_value(
        document,
        ("export_to_json", "model_dump_json", "json"),
        warnings,
    )
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            warnings.append(f"Docling JSON export {method_name} was not valid JSON: {exc}.")
            return method_name, None
        if isinstance(loaded, dict):
            return method_name, loaded
    return method_name, None


def dict_label(node):
    label = node.get("label") or node.get("type") or node.get("content_type") or "text"
    if isinstance(label, dict):
        label = label.get("value") or label.get("name") or "text"
    return str(label).lower()


def dict_page_no(node):
    for key in ("page_no", "page", "page_number"):
        value = node.get(key)
        if isinstance(value, int):
            return value
    provenance = node.get("prov") or node.get("provenance")
    if isinstance(provenance, list) and provenance:
        first = provenance[0]
        if isinstance(first, dict):
            return dict_page_no(first)
    return None


def dict_bbox(node):
    value = node.get("bbox") or node.get("bounding_box")
    if not isinstance(value, dict):
        return None
    x0 = value.get("x0", value.get("l", value.get("left")))
    y0 = value.get("y0", value.get("t", value.get("top")))
    x1 = value.get("x1", value.get("r", value.get("right")))
    y1 = value.get("y1", value.get("b", value.get("bottom")))
    if all(item is None for item in (x0, y0, x1, y1)):
        return None
    return BoundingBox(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        coord_origin=value.get("coord_origin"),
    )


def iter_dict_text_nodes(value):
    if isinstance(value, dict):
        text = value.get("text") or value.get("content")
        if isinstance(text, str) and text.strip():
            yield value, text
        for child in value.values():
            yield from iter_dict_text_nodes(child)
    elif isinstance(value, list):
        for item in value:
            yield from iter_dict_text_nodes(item)


def extract_dict_blocks(document, document_id, page_total, warnings, asset_store=None):
    method_name, exported = export_dict(document, warnings)
    if not exported:
        warnings.append("Docling dict/json export unavailable or empty.")
        return [], None

    blocks = []
    seen = set()
    for node, text in iter_dict_text_nodes(exported):
        normalized = normalize_plain_text(text)
        if not normalized:
            continue
        page_no = dict_page_no(node)
        label = dict_label(node)
        metadata = {}
        if label in VISUAL_LABELS:
            asset = None
            if asset_store is not None:
                asset = asset_store.save_figure(
                    normalized,
                    page_no,
                    len([block for block in blocks if block.metadata.get("asset_type") == "image"]) + 1,
                    metadata={"source_label": label},
                )
            metadata.update(make_asset_metadata(asset))
            normalized = ""
        key = (page_no, label, normalized)
        if key in seen:
            continue
        seen.add(key)
        blocks.append(
            make_block(
                document_id=document_id,
                page_total=page_total,
                sequence_no=len(blocks) + 1,
                text=normalized,
                page_no=page_no,
                label=label,
                bbox=dict_bbox(node),
                metadata=metadata,
            )
        )
    if not blocks:
        warnings.append(f"Docling dict/json export {method_name} produced no text blocks.")
    return blocks, method_name


def extract_markdown_blocks(document, document_id, page_total, warnings, asset_store=None):
    method_name, markdown = public_export_value(
        document,
        ("export_to_markdown", "to_markdown"),
        warnings,
    )
    if not isinstance(markdown, str) or not markdown.strip():
        warnings.append("Docling markdown export unavailable or empty.")
        return [], None

    blocks = []
    for paragraph in markdown.split("\n\n"):
        text = normalize_plain_text(paragraph)
        if not text:
            continue
        data, _ = decode_data_uri(text)
        label = "figure" if data is not None else "section_header" if text.startswith("#") else "text"
        metadata = {}
        if label == "figure":
            asset = None
            if asset_store is not None:
                asset = asset_store.save_figure(
                    text,
                    None,
                    len([block for block in blocks if block.metadata.get("asset_type") == "image"]) + 1,
                    metadata={"source_label": label},
                )
            metadata.update(make_asset_metadata(asset))
            text = ""
        blocks.append(
            make_block(
                document_id=document_id,
                page_total=page_total,
                sequence_no=len(blocks) + 1,
                text=text.lstrip("#").strip(),
                label=label,
                markdown=text if label != "figure" else None,
                metadata=metadata,
            )
        )
    if not blocks:
        warnings.append(f"Docling markdown export {method_name} produced no text blocks.")
    return blocks, method_name


def document_assets(blocks):
    assets = []
    seen = set()
    for block in blocks:
        asset = document_asset_from_figure_block(block)
        if asset is None:
            continue
        key = asset.asset_id or asset.asset_uri
        if key in seen:
            continue
        seen.add(key)
        assets.append(asset)
    return assets


def docling_document_to_parsed_document(document, source_path, asset_store=None):
    source = Path(source_path)
    document_id = document_id_for_path(source)
    page_total = len(getattr(document, "pages", []))
    warnings = []
    extraction_strategy = None

    blocks = extract_native_blocks(document, document_id, page_total, warnings, asset_store=asset_store)
    if blocks:
        extraction_strategy = "native_iterate_items"
    else:
        blocks, method_name = extract_dict_blocks(
            document,
            document_id,
            page_total,
            warnings,
            asset_store=asset_store,
        )
        if blocks:
            extraction_strategy = "docling_dict_export"
            warnings.append(f"Docling fallback used dict/json export method: {method_name}.")

    if not blocks:
        blocks, method_name = extract_markdown_blocks(
            document,
            document_id,
            page_total,
            warnings,
            asset_store=asset_store,
        )
        if blocks:
            extraction_strategy = "docling_markdown_export"
            warnings.append(f"Docling fallback used markdown export method: {method_name}.")

    if not blocks:
        extraction_strategy = "none"
        warnings.append("Docling adapter produced no blocks after native, dict/json, and markdown extraction.")

    assets = document_assets(blocks)
    return ParsedDocument(
        document_id=document_id,
        source_path=str(source),
        source_name=source.name,
        page_total=page_total,
        blocks=blocks,
        assets=assets,
        metadata={
            "parser_name": "docling",
            "docling_version": installed_docling_version(),
            "extraction_strategy": extraction_strategy,
            "assets": [to_dict(asset) for asset in assets],
            "warnings": warnings,
        },
    )
