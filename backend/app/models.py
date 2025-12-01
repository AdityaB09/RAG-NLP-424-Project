from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
from uuid import uuid4
from datetime import datetime


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_number: int
    index_in_doc: int


@dataclass
class Document:
    doc_id: str
    title: str
    source_type: str
    topics: List[str]
    num_chunks: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QAlog:
    """
    One row per /api/rag/query call – used for dashboard + evaluation.
    """
    log_id: str
    timestamp: datetime
    question: str
    mode: str
    top_k: int
    rerank: bool
    used_docs: List[str]
    grounded: bool
    answerability: str  # HIGH / MEDIUM / LOW
    refused: bool

    retrieval_ms: float
    generation_ms: float
    total_ms: float


# These are only used inside concept-graph building
@dataclass
class ConceptNode:
    id: str
    label: str
    type: str  # "document" or "concept"


@dataclass
class ConceptEdge:
    source: str
    target: str
    weight: float = 1.0


def new_log_id() -> str:
    return str(uuid4())
