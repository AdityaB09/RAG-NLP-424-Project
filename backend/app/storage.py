from __future__ import annotations
from typing import Dict, List, Optional
from uuid import uuid4

from .models import Document, Chunk, QAlog


class InMemoryStore:
    def __init__(self) -> None:
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}
        self.chunks_by_doc: Dict[str, List[str]] = {}
        self.logs: List[QAlog] = []

        # Vectorizer & matrix for BM25-like retrieval
        self.vectorizer = None
        self.tfidf_matrix = None
        self.chunk_id_to_index: Dict[str, int] = {}

    def add_document(self, doc: Document) -> None:
        self.documents[doc.doc_id] = doc
        self.chunks_by_doc.setdefault(doc.doc_id, [])

    def add_chunk(self, chunk: Chunk) -> None:
        self.chunks[chunk.chunk_id] = chunk
        self.chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk.chunk_id)

    def add_log(self, log: QAlog) -> None:
        self.logs.append(log)

    def next_log_id(self) -> str:
        return str(uuid4())


store = InMemoryStore()
