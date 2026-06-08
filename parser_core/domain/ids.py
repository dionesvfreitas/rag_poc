import hashlib
from pathlib import Path


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_id_for_path(path: str | Path) -> str:
    source = Path(path)
    stat = source.stat()
    file_hash = file_sha256(source)
    return sha256_text(f"{source.resolve()}:{stat.st_size}:{file_hash}")


def block_id(document_id: str, page_no: int | None, sequence_no: int) -> str:
    return sha256_text(f"{document_id}:{page_no}:{sequence_no}")


def chunk_id(document_id: str, chunk_no: int) -> str:
    return sha256_text(f"{document_id}:{chunk_no}")
