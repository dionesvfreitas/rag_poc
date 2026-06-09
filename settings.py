import os
from dataclasses import dataclass


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_INDEX_PATH = "index/document_index.json"
DEFAULT_TOP_K = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.0


@dataclass(frozen=True)
class Settings:
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = DEFAULT_TOP_K
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    index_path: str = DEFAULT_INDEX_PATH

    @classmethod
    def from_env(cls):
        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            top_k=int(os.getenv("TOP_K", str(DEFAULT_TOP_K))),
            similarity_threshold=float(
                os.getenv("SIMILARITY_THRESHOLD", str(DEFAULT_SIMILARITY_THRESHOLD))
            ),
            index_path=os.getenv("INDEX_PATH", DEFAULT_INDEX_PATH),
        )

