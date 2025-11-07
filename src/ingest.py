import os
from typing import List
from src.utils.pdf_loader import load_pdf_text, split_text_to_chunks
from src.utils.vectorstore import VectorStore
from langchain.schema import Document
from tqdm import tqdm

def ingest_files(file_paths: List[str], source_name: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> dict:
    """
    Ingests files (PDF/TXT) into vectorstore.
    Returns a dict with ingested file names and total chunk count.
    """
    vs = VectorStore()
    all_docs = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            text = load_pdf_text(path)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        chunks = split_text_to_chunks(text, chunk_size=chunk_size, overlap=chunk_overlap)
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": source_name or os.path.basename(path),
                "path": path,
                "chunk_index": i
            }
            all_docs.append(Document(page_content=chunk, metadata=metadata))
    if all_docs:
        vs.add_documents(all_docs)
    return {"ingested_files": [os.path.basename(p) for p in file_paths], "total_chunks": len(all_docs)}