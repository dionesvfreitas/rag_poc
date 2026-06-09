from abc import ABC, abstractmethod

from settings import DEFAULT_EMBEDDING_MODEL


class EmbeddingProvider(ABC):
    model_name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector for each input text."""


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name=DEFAULT_EMBEDDING_MODEL, model=None, normalize_embeddings=True):
        self.model_name = model_name
        self._model = model
        self.normalize_embeddings = normalize_embeddings

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required to build or query embeddings. "
                    "Install project dependencies with: python -m pip install -r requirements.txt"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        return [[float(value) for value in embedding] for embedding in embeddings]

