# MedRAG

Lightweight Retrieval-Augmented Generation (RAG) project skeleton for medical/clinical document assistance.

Project layout
# MedRAG

Lightweight Retrieval-Augmented Generation (RAG) project skeleton focused on medical/clinical document assistance. It demonstrates ingestion, vector storage, retrieval, and LLM-based report generation plus evaluation tooling.

Contents

```
MedRAG/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ src/
│  ├─ config.py            # configuration & env loading
│  ├─ ingest.py            # ingestion pipeline (PDF -> docs -> embeddings)
│  ├─ report_generator.py  # core retrieval + LLM report generation
│  ├─ prompts.py           # prompt templates used for generation
│  ├─ eval/                # evaluation scripts and data (queries/qrels)
│  └─ utils/               # helpers: embeddings, vectorstore, pdf loader, web search
└─ data/                   # runtime data and vectorstore persistence
```

Quick start (local development)

1. Create and activate a Python virtualenv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file from the example and set any required API keys:

```powershell
copy .env.example .env
# then edit .env with your keys (do not commit .env)
```

4. Run a small test report generator (quick smoke test):

```powershell
python run_report_test.py
```

This script seeds the vectorstore (if empty) and calls `ReportGenerator.generate()` to produce a sample clinical report printed to stdout.

Evaluation & batch reports

Use the evaluation scripts under `src/eval` to run batch retrieval/reporting and compute metrics (MAP, NDCG, MRR, precision/recall). Example:

```powershell
# from repo root
cd "C:\ai project\MedRAG"
python -m src.eval.run_batch_reports --queries src/eval/queries_multi.jsonl --qrels src/eval/qrels_multi_graded.tsv --out-dir src/eval/batch_reports_graded --k 3
```

Key files

- `src/report_generator.py` — orchestrates retrieval + LLM calls to make structured clinical reports.
- `src/utils/vectorstore.py` — local vectorstore abstraction (uses FAISS via LangChain when available; otherwise uses local embeddings and dot-product retrieval).
- `src/utils/embeddings.py` — embedding client wrapper for local/cloud providers.
- `src/models/openai_client.py` — LLM API wrapper (OpenAI/Groq/other abstractions).
- `src/eval/run_batch_reports.py` — runs queries, generates reports, and writes per-query JSON reports + batch metrics.

Repository & pushing

GitHub repo: https://github.com/Muqtarali/MedRAG (this repo was pushed from the local copy).

To push locally (example):

```powershell
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/Muqtarali/MedRAG.git
git branch -M main
git push -u origin main
```

Security & housekeeping

- `.env` is ignored (`.gitignore`) — do not commit secrets.
- Consider removing generated evaluation outputs (folders under `src/eval/*/reports`) from version control and adding them to `.gitignore` to keep the repo small.
- For large binaries like PPTX consider Git LFS.

Development notes

- The local vector store uses dot-product search by default; normalize embeddings (or configure FAISS IndexFlatIP) for cosine similarity if desired — see `src/utils/vectorstore.py`.
- The code is arranged to be easy to swap components (replace embedding client, switch to production vectorstore, or replace LLM client).

Contributing

1. Fork and branch.
2. Run tests / linters (none included by default).
3. Open a PR with a clear description.

License

This repository does not include a license file. Add `LICENSE` if you want to publish under a specific license.

If you want a shorter README, more detail in any section, or a README translated to another format, tell me which part to adjust and I'll update it.
