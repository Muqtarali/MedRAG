# MedRAG

Lightweight Retrieval-Augmented Generation (RAG) project skeleton for medical/clinical document assistance.

Project layout

```
MedRAG/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ schemas.py
│  ├─ main.py
│  ├─ ingest.py
│  ├─ report_generator.py
│  ├─ prompts.py
│  ├─ cli_demo.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ openai_client.py
│  └─ utils/
│     ├─ __init__.py
     ├─ pdf_loader.py
     ├─ embeddings.py
     └─ vectorstore.py
└─ data/                (created at runtime / vectorstore persistence)
```

Quick start

1. Create a Python virtual environment and activate it.
2. Install minimal requirements:

   pip install -r requirements.txt

3. Edit `.env` or copy `.env.example` and set provider keys (GROQ/OLLAMA/custom).
4. Populate `data/books/` with PDFs and run the demo:

   python -m src.main

Notes
- This is a minimal starter: embedding/vectorstore/LLM code are lightweight stubs intended for local dev and testing. Replace with production components (FAISS/Chroma/embedding service) as needed.

Required environment variables

The project will load environment variables from a local `.env` file (call `load_dotenv()` from `src/config.py`). Create a `.env` at the project root or set the variables in your shell. Common variables used by the demo:

- `GROQ_API_URL` — Groq OpenAI-compatible endpoint (example: `https://api.groq.com/openai/v1/chat/completions`)
- `GROQ_API_KEY` — Your Groq API key (secret)
- `GROQ_MODEL` (optional) — Model identifier to use (overrides defaults in the runner)
- `GOOGLE_API_KEY` — Google Cloud API key for Programmable Search (Custom Search JSON API)
- `GOOGLE_CX` — Google Custom Search Engine id (cx)

Make sure `.env` is listed in `.gitignore` so you don't commit secrets.
