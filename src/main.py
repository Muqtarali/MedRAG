from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import shutil
import os
from src.ingest import ingest_files
from src.report_generator import ReportGenerator
from src.schemas import IngestResponse, QueryRequest, QueryResponse, RetrievedChunk
from src.config import settings

app = FastAPI(title="MedRAG: LLM-Powered Diagnostic Report Generation")

TEMP_UPLOAD_DIR = "./tmp_uploads"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(files: List[UploadFile] = File(...), source_name: str = None):
    """
    Upload PDFs or text files to ingest into the vector store.
    """
    saved_paths = []
    try:
        for f in files:
            dest = os.path.join(TEMP_UPLOAD_DIR, f.filename)
            with open(dest, "wb") as out_f:
                content = await f.read()
                out_f.write(content)
            saved_paths.append(dest)
        result = ingest_files(saved_paths, source_name=source_name, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """
    Query the vector store + LLM to generate diagnostic report.
    """
    try:
        gen = ReportGenerator()
        report, retrieved = gen.generate(patient=req.patient.dict(), question=req.question, top_k=req.top_k, llm_model=req.llm_model)
        retrieved_serializable = []
        for r in retrieved:
            retrieved_serializable.append(RetrievedChunk(content=r.page_content, metadata=r.metadata))
        return QueryResponse(report=report, retrieved=retrieved_serializable)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))