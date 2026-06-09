class CitationBuilder:
    def build(self, search_results):
        citations = []
        for result in search_results:
            metadata = result.get("metadata") or {}
            page_start = metadata.get("page_start")
            page_end = metadata.get("page_end")
            citations.append(
                {
                    "chunk_id": result.get("chunk", {}).get("chunk_id"),
                    "source_chunk_id": metadata.get("source_chunk_id"),
                    "pages": pages_from_range(page_start, page_end),
                    "page_start": page_start,
                    "page_end": page_end,
                    "section_path": list(metadata.get("section_path") or []),
                    "source_block_ids": list(metadata.get("source_block_ids") or []),
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

