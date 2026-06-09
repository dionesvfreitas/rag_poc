from mini_rag.embeddings import EmbeddingProvider
from mini_rag.store import LocalVectorStore
from settings import Settings


class Retriever:
    def __init__(
        self,
        vector_store: LocalVectorStore,
        embedding_provider: EmbeddingProvider,
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ):
        settings = Settings.from_env()
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.top_k = top_k if top_k is not None else settings.top_k
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )

    def search(self, query: str, top_k: int | None = None):
        query_embedding = self.embedding_provider.embed([query])[0]
        return self.vector_store.search(
            query_embedding,
            top_k=top_k if top_k is not None else self.top_k,
            similarity_threshold=self.similarity_threshold,
        )

