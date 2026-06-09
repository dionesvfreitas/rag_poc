import base64
import re
from pathlib import Path
from typing import Any

from parser_core.domain.models import DocumentAsset
from parser_core.domain.models import to_dict


DATA_URI_RE = re.compile(
    r"data:(?P<mime>image/[a-zA-Z0-9.+-]+);base64,(?P<data>[A-Za-z0-9+/=\s]+)"
)
EXTENSION_BY_MIME = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def image_extension(mime_type):
    return EXTENSION_BY_MIME.get((mime_type or "").lower(), ".png")


def decode_data_uri(value):
    if not isinstance(value, str):
        return None, None
    match = DATA_URI_RE.search(value)
    if not match:
        return None, None
    try:
        return base64.b64decode(match.group("data"), validate=False), match.group("mime").lower()
    except ValueError:
        return None, None


def asset_record(asset):
    return to_dict(asset) if hasattr(asset, "__dataclass_fields__") else dict(asset)


class LocalAssetStore:
    def __init__(self, root_path="outputs/images", uri_prefix="images"):
        self.root_path = Path(root_path)
        self.uri_prefix = uri_prefix.strip("/")

    def figure_asset_id(self, sequence_no):
        return f"fig_{sequence_no:03d}"

    def figure_file_name(self, page_no, sequence_no, extension=".png"):
        page = page_no if page_no is not None else 0
        return f"page_{page:03d}_figure_{sequence_no:03d}{extension}"

    def asset_uri(self, file_name):
        return f"{self.uri_prefix}/{file_name}" if self.uri_prefix else file_name

    def save_figure(self, image_source, page_no, sequence_no, metadata=None):
        metadata = dict(metadata or {})
        data, mime_type = image_bytes(image_source)
        if data is None:
            return None

        asset_id = self.figure_asset_id(sequence_no)
        extension = image_extension(mime_type)
        file_name = self.figure_file_name(page_no, sequence_no, extension)
        self.root_path.mkdir(parents=True, exist_ok=True)
        (self.root_path / file_name).write_bytes(data)
        return DocumentAsset(
            asset_id=asset_id,
            asset_type="image",
            asset_uri=self.asset_uri(file_name),
            page_no=page_no,
            metadata={
                **metadata,
                "mime_type": mime_type or "image/png",
                "storage_backend": "local",
            },
        )


def image_bytes(image_source: Any):
    data, mime_type = decode_data_uri(image_source)
    if data is not None:
        return data, mime_type

    if isinstance(image_source, bytes):
        return image_source, None

    for attr_name in ("data", "bytes", "content"):
        value = getattr(image_source, attr_name, None)
        if isinstance(value, bytes):
            return value, getattr(image_source, "mime_type", None)

    for attr_name in ("pil_image", "image"):
        value = getattr(image_source, attr_name, None)
        if value is not None and value is not image_source:
            data, mime_type = image_bytes(value)
            if data is not None:
                return data, mime_type

    if hasattr(image_source, "save"):
        from io import BytesIO

        output = BytesIO()
        image_source.save(output, format="PNG")
        return output.getvalue(), "image/png"

    return None, None
