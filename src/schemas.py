from typing import Optional, List, Dict
from pydantic import BaseModel

class IngestResponse(BaseModel):
    ingested_files: List[str]
    total_chunks: int

class PatientInfo(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    history: Optional[str] = None

class QueryRequest(BaseModel):
    patient: PatientInfo
    question: str
    top_k: Optional[int] = 6
    llm_model: Optional[str] = None

class RetrievedChunk(BaseModel):
    content: str
    metadata: Dict

class QueryResponse(BaseModel):
    report: str
    retrieved: List[RetrievedChunk]