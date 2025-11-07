import os
from typing import List, Tuple
from langchain.schema import Document
from src.utils.embeddings import EmbeddingClient
from src.config import settings
import os
import numpy as np


class VectorStore:
    """Wrapper that uses FAISS/langchain when cloud embeddings are available,
    otherwise uses an in-memory TF-IDF-backed store for local demos."""
    def __init__(self, persist_path: str = None):
        self.persist_path = persist_path or settings.VECTORSTORE_PATH
        self.embedding_client = EmbeddingClient()
        self._docs: List[Document] = []
        self._embs = None
        self._is_local = self.embedding_client.provider == "local"
        # If using a non-local (FAISS) store, try to load an existing persisted store
        if not self._is_local:
            try:
                from langchain.vectorstores import FAISS
                # load_local may raise if path not present; guard with exists
                if os.path.exists(self.persist_path):
                    try:
                        self.store = FAISS.load_local(self.persist_path, embeddings=self._get_langchain_embeddings())
                    except Exception:
                        # fallback: no loaded store
                        self.store = None
                else:
                    self.store = None
            except Exception:
                # langchain/FAISS not available
                self.store = None

    def add_documents(self, docs: List[Document]):
        # store docs and compute embeddings
        texts = [d.page_content for d in docs]
        if self._is_local:
            mat = self.embedding_client.embed_documents(texts)
            # mat is a sparse matrix
            if self._embs is None:
                self._embs = mat
            else:
                from scipy.sparse import vstack
                self._embs = vstack([self._embs, mat])
            base_idx = len(self._docs)
            for i, d in enumerate(docs):
                self._docs.append(d)
        else:
            # For cloud-backed embeddings we defer to langchain FAISS store
            try:
                from langchain.vectorstores import FAISS
                from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
            except Exception:
                raise RuntimeError("FAISS/langchain not available in this environment")
            if getattr(self, "store", None) is None:
                self.store = FAISS.from_documents(docs, embeddings=self._get_langchain_embeddings())
            else:
                self.store.add_documents(docs)
            os.makedirs(self.persist_path, exist_ok=True)
            try:
                self.store.save_local(self.persist_path)
            except Exception:
                pass

    def _get_langchain_embeddings(self):
        if self.embedding_client.provider == "openai":
            from langchain.embeddings import OpenAIEmbeddings
            return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        else:
            from langchain.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=settings.HF_EMBEDDING_MODEL)

    def similarity_search_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if self._is_local:
            if self._embs is None or len(self._docs) == 0:
                return []
            qv = self.embedding_client.embed_query(query)
            # compute dot-product similarity
            sims = (self._embs @ qv.T).toarray().ravel()
            idxs = np.argsort(sims)[::-1][:k]
            return [(self._docs[int(i)], float(sims[int(i)])) for i in idxs]
        else:
            if getattr(self, "store", None) is None:
                return []
            return self.store.similarity_search_with_score(query, k=k)

    def is_empty(self) -> bool:
        if self._is_local:
            return len(self._docs) == 0
        return getattr(self, "store", None) is None