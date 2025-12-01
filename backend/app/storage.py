from __future__ import annotations
from typing import Dict, List

from .models import Document, Chunk, QAlog


class InMemoryStore:
    """
    Very simple in-memory store.
    Good enough for your demo / presentation.
    """

    def __init__(self) -> None:
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}
        self.chunks_by_doc: Dict[str, List[str]] = {}
        self.logs: List[QAlog] = []

        # Vectorizer & matrix for BM25-ish retrieval
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


store = InMemoryStore()
