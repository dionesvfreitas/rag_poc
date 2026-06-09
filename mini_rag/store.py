import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass
class VectorItem:
    chunk_id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class LocalVectorStore:
    def __init__(
        self,
        items: list[VectorItem] | None = None,
        *,
        embedding_model: str | None = None,
        source_path: str | None = None,
        schema_version: str = SCHEMA_VERSION,
    ):
        self.items = list(items or [])
        self.embedding_model = embedding_model
        self.source_path = source_path
        self.schema_version = schema_version

    def add(self, chunk_id: str, content: str, embedding: list[float], metadata=None):
        self.items.append(
            VectorItem(
                chunk_id=chunk_id,
                content=content,
                embedding=[float(value) for value in embedding],
                metadata=dict(metadata or {}),
            )
        )

    def search(self, query_embedding: list[float], top_k=10, similarity_threshold=None):
        scored = []
        for item in self.items:
            score = cosine_similarity(query_embedding, item.embedding)
            if similarity_threshold is not None and score < similarity_threshold:
                continue
            scored.append(
                {
                    "score": score,
                    "chunk": {
                        "chunk_id": item.chunk_id,
                        "content": item.content,
                    },
                    "metadata": dict(item.metadata),
                }
            )
        scored.sort(key=lambda result: result["score"], reverse=True)
        return scored[:top_k]

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "embedding_model": self.embedding_model,
            "source_path": self.source_path,
            "items": [asdict(item) for item in self.items],
        }

    def save(self, path):
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = [
            VectorItem(
                chunk_id=item["chunk_id"],
                content=item.get("content") or "",
                embedding=[float(value) for value in item.get("embedding") or []],
                metadata=dict(item.get("metadata") or {}),
            )
            for item in data.get("items") or []
        ]
        return cls(
            items,
            embedding_model=data.get("embedding_model"),
            source_path=data.get("source_path"),
            schema_version=data.get("schema_version") or SCHEMA_VERSION,
        )

