from typing import List
from src.config import settings

try:
    # Prefer using LangChain wrappers when a cloud key is configured
    from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
except Exception:
    OpenAIEmbeddings = None
    HuggingFaceEmbeddings = None

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class EmbeddingClient:
    """Flexible embedding client.

    - If settings.EMBEDDING_PROVIDER == 'openai' and OpenAIEmbeddings is
      available, uses that (requires OPENAI_API_KEY).
    - Else if provider == 'hf' and HuggingFaceEmbeddings available, uses that.
    - Otherwise falls back to a local TF-IDF embedder (fast, demo-only).
    """
    def __init__(self):
        provider = settings.EMBEDDING_PROVIDER.lower()
        self.provider = "local"
        self._vectorizer = None
        # try cloud providers when requested and available
        if provider == "openai" and OpenAIEmbeddings is not None:
            try:
                self._client = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
                self.provider = "openai"
                return
            except Exception:
                self._client = None
        if provider == "hf" and HuggingFaceEmbeddings is not None:
            try:
                self._client = HuggingFaceEmbeddings(model_name=settings.HF_EMBEDDING_MODEL)
                self.provider = "hf"
                return
            except Exception:
                self._client = None

        # Fallback local TF-IDF
        self._client = None
        self.provider = "local"

    def embed_documents(self, texts: List[str]):
        if self.provider in ("openai", "hf") and self._client is not None:
            return self._client.embed_documents(texts)
        # local tf-idf dense vectors
        self._vectorizer = TfidfVectorizer().fit(texts)
        mat = self._vectorizer.transform(texts)
        return mat

    def embed_query(self, text: str):
        if self.provider in ("openai", "hf") and self._client is not None:
            return self._client.embed_query(text)
        if self._vectorizer is None:
            # client must be fit on documents first
            raise RuntimeError("Local embedder not fit; call embed_documents first")
        return self._vectorizer.transform([text])
