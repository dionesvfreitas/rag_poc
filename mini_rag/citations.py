class CitationBuilder:
    def build(self, search_results):
        citations = []
        for result in search_results:
            metadata = result.get("metadata") or {}
            page_start = value_from_result(result, metadata, "page_start")
            page_end = value_from_result(result, metadata, "page_end")
            citations.append(
                {
                    "chunk_id": chunk_id_from_result(result),
                    "source_chunk_id": value_from_result(result, metadata, "source_chunk_id"),
                    "pages": pages_from_range(page_start, page_end),
                    "page_start": page_start,
                    "page_end": page_end,
                    "section_path": list(value_from_result(result, metadata, "section_path") or []),
                    "source_block_ids": list(
                        value_from_result(result, metadata, "source_block_ids") or []
                    ),
                    "source_spans": list(value_from_result(result, metadata, "source_spans") or []),
                    "related_assets": list(
                        value_from_result(result, metadata, "related_assets") or []
                    ),
                    "score": result.get("score"),
                }
            )
        return citations


def pages_from_range(page_start, page_end):
    if not isinstance(page_start, int) and not isinstance(page_end, int):
        return []
    if isinstance(page_start, int) and not isinstance(page_end, int):
        return [page_start]
    if isinstance(page_end, int) and not isinstance(page_start, int):
        return [page_end]
    if page_end < page_start:
        return [page_start]
    return list(range(page_start, page_end + 1))


def chunk_id_from_result(result):
    chunk_id = result.get("chunk_id")
    if chunk_id is not None:
        return chunk_id
    return result.get("chunk", {}).get("chunk_id")


def value_from_result(result, metadata, field):
    if field in metadata:
        return metadata.get(field)
    return result.get(field)
