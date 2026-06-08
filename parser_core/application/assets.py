from parser_core.domain.models import AssetType, DocumentAsset


def document_asset_from_figure_block(block):
    metadata = dict(block.metadata or {})
    asset_data = dict(metadata.get("asset") or {})
    asset_id = metadata.get("asset_id") or asset_data.get("asset_id")
    asset_uri = metadata.get("asset_uri") or asset_data.get("asset_uri")
    if not asset_id or not asset_uri:
        return None

    asset_metadata = dict(asset_data.get("metadata") or {})
    asset_metadata.update(metadata.get("asset_metadata") or {})
    if metadata.get("label"):
        asset_metadata.setdefault("source_label", metadata["label"])
    if metadata.get("asset_link_status"):
        asset_metadata["asset_link_status"] = metadata["asset_link_status"]

    return DocumentAsset(
        asset_id=asset_id,
        asset_uri=asset_uri,
        asset_type=metadata.get("asset_type") or asset_data.get("asset_type") or AssetType.UNKNOWN,
        page_no=asset_data.get("page_no", block.page_no),
        source_block_id=block.block_id,
        bbox=block.bbox,
        caption=metadata.get("caption"),
        ocr_text=metadata.get("ocr_text"),
        metadata=asset_metadata,
    )
