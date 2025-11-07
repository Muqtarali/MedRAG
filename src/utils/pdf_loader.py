from typing import List
import fitz  # PyMuPDF

def load_pdf_text(path: str) -> str:
    """
    Extracts text from a PDF using PyMuPDF.
    Returns the full text as a single string.
    """
    doc = fitz.open(path)
    texts = []
    for page in doc:
        text = page.get_text("text")
        if text:
            texts.append(text)
    doc.close()
    return "\n".join(texts)

def split_text_to_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits a long text into overlapping chunks for embedding.
    """
    chunks = []
    start = 0
    text_len = len(text)
    if text_len == 0:
        return []
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        # advance start: keep overlap
        start = end - overlap
        if start < 0:
            start = 0
        if start >= text_len:
            break
    return chunks