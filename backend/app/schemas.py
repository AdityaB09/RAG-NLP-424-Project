from __future__ import annotations
from typing import List, Optional, Dict
from pydantic import BaseModel


# ---------- Documents ----------


class DocumentOut(BaseModel):
    doc_id: str
    title: str
    source_type: str
    topics: List[str]
    num_chunks: int
    created_at: str
    updated_at: str


class DocumentListOut(BaseModel):
    documents: List[DocumentOut]


class DocUploadResponse(BaseModel):
    doc_id: str
    title: str
    num_chunks: int


# ---------- RAG Query ----------


class Citation(BaseModel):
    doc_id: str
    doc_title: str
    page_number: int
    snippet: str
    score_bm25: float
    score_dense: float
    score_final: float


class RAGQueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"  # "bm25" | "dense" | "hybrid"
    top_k: int = 5
    rerank: bool = True


class RAGQueryResponse(BaseModel):
    answer: str
    answerability: str
    refused: bool
    reason: Optional[str] = None
    citations: List[Citation]
    retrieval_graph: Dict
    timings_ms: Dict[str, float]


# ---------- Overview / Dashboard ----------


class DayCount(BaseModel):
    day: str
    count: int


class OverviewStats(BaseModel):
    num_documents: int
    num_chunks: int
    num_questions: int
    grounded_ratio: float
    mode_counts: Dict[str, int]
    questions_over_time: List[DayCount]


# ---------- Logs ----------


class LogItem(BaseModel):
    log_id: str
    timestamp: str
    question: str
    mode: str
    top_k: int
    rerank: bool
    used_docs: List[str]
    grounded: bool
    answerability: str
    refused: bool
    total_ms: float


class LogsResponse(BaseModel):
    logs: List[LogItem]


# ---------- Concept graph ----------


class ConceptNodeSchema(BaseModel):
    id: str
    label: str
    type: str  # "document" | "concept"


class ConceptEdgeSchema(BaseModel):
    source: str
    target: str
    weight: float


class ConceptGraphResponse(BaseModel):
    nodes: List[ConceptNodeSchema]
    edges: List[ConceptEdgeSchema]


# ---------- Freshness ----------


class FreshnessQuestion(BaseModel):
    question: str
    count: int


class FreshnessExample(BaseModel):
    timestamp: str
    used_docs: List[str]
    grounded: bool
    answerability: str
    refused: bool
    total_ms: float


class FreshnessResponse(BaseModel):
    question: str
    examples: List[FreshnessExample]
