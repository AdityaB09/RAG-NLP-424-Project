from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
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
    log_id: str
    timestamp: datetime
    question: str
    mode: str
    used_docs: List[str]
    grounded: bool
    answerability: str


@dataclass
class ConceptNode:
    id: str
    label: str
    type: str  # "concept" or "document"


@dataclass
class ConceptEdge:
    source: str
    target: str
    weight: float = 1.0
