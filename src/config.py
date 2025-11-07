import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")  # openai|hf
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai|hf
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./data/faiss_store")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    # Web search / external context
    BING_API_KEY = os.getenv("BING_API_KEY", "")
    BING_ENDPOINT = os.getenv("BING_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
    # Google Programmable Search (Custom Search JSON API)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CX = os.getenv("GOOGLE_CX", "")

settings = Settings()