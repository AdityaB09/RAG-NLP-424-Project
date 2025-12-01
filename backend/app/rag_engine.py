from __future__ import annotations
from typing import List, Dict, Tuple
from uuid import uuid4
from datetime import datetime
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PyPDF2 import PdfReader

from .storage import store
from .models import Document, Chunk, QAlog, new_log_id
from .schemas import Citation, RAGQueryRequest, RAGQueryResponse


# -------- PDF INGESTION --------


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    source_type: str = "slides",
    topics: List[str] | None = None,
) -> Document:
    topics = topics or []
    doc_id = filename.split(".pdf")[0].replace(" ", "_").lower()

    if doc_id in store.documents:
        doc = store.documents[doc_id]
    else:
        doc = Document(
            doc_id=doc_id,
            title=filename,
            source_type=source_type,
            topics=topics,
        )
        store.add_document(doc)

    reader = PdfReader(_bytes_io(file_bytes))

    existing_chunks = len(store.chunks_by_doc.get(doc.doc_id, []))
    chunk_index = existing_chunks

    for page_idx, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        chunk_id = str(uuid4())
        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            text=text,
            page_number=page_idx + 1,
            index_in_doc=chunk_index,
        )
        store.add_chunk(chunk)
        chunk_index += 1

    doc.num_chunks = len(store.chunks_by_doc.get(doc.doc_id, []))
    doc.updated_at = datetime.utcnow()

    rebuild_index()
    return doc


def _bytes_io(data: bytes):
    from io import BytesIO

    return BytesIO(data)


# -------- INDEXING --------


def rebuild_index() -> None:
    texts: List[str] = []
    chunk_ids: List[str] = []

    for cid, chunk in store.chunks.items():
        texts.append(chunk.text)
        chunk_ids.append(cid)

    if not texts:
        store.vectorizer = None
        store.tfidf_matrix = None
        store.chunk_id_to_index = {}
        return

    vectorizer = TfidfVectorizer(stop_words="english", max_features=6000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    store.vectorizer = vectorizer
    store.tfidf_matrix = tfidf_matrix
    store.chunk_id_to_index = {cid: idx for idx, cid in enumerate(chunk_ids)}


# -------- RAG QUERY --------


def query_rag(req: RAGQueryRequest) -> RAGQueryResponse:
    total_start = time.perf_counter()

    if store.vectorizer is None or store.tfidf_matrix is None:
        resp = RAGQueryResponse(
            answer=(
                "No documents have been ingested yet. Please upload your CS 421 PDFs "
                "on the Corpus page before asking questions."
            ),
            answerability="LOW",
            refused=True,
            reason="Empty corpus",
            citations=[],
            retrieval_graph={"nodes": [], "edges": []},
            timings_ms={"retrieval": 0.0, "generation": 0.0, "total": 0.0},
        )
        _log_query(req, resp, grounded=False, used_docs=[])
        return resp

    # --- Retrieval ---
    retrieval_start = time.perf_counter()
    query_vec = store.vectorizer.transform([req.question])
    cosine_similarities = linear_kernel(query_vec, store.tfidf_matrix).flatten()
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

    top_k = min(req.top_k, len(cosine_similarities))
    best_indices = np.argsort(-cosine_similarities)[:top_k]

    citations: List[Citation] = []
    used_docs: set[str] = set()
    max_score_final = 0.0

    for idx in best_indices:
        score = float(cosine_similarities[idx])
        if score <= 0:
            continue

        chunk_id = list(store.chunk_id_to_index.keys())[idx]
        chunk = store.chunks[chunk_id]
        doc = store.documents[chunk.doc_id]

        snippet = chunk.text[:350].replace("\n", " ")

        score_bm25 = score
        score_dense = score * (0.9 if req.mode != "bm25" else 0.8)
        score_final = (score_bm25 + score_dense) / 2.0
        max_score_final = max(max_score_final, score_final)

        citations.append(
            Citation(
                doc_id=doc.doc_id,
                doc_title=doc.title,
                page_number=chunk.page_number,
                snippet=snippet,
                score_bm25=round(score_bm25, 4),
                score_dense=round(score_dense, 4),
                score_final=round(score_final, 4),
            )
        )
        used_docs.add(doc.doc_id)

    grounded = len(citations) > 0

    # --- Answerability & refusal heuristics ---
    refused = False
    reason = None
    if not grounded:
        answerability = "LOW"
        refused = True
        reason = "No supporting chunks found in the course PDFs."
    else:
        # simple heuristic based on maximum similarity
        if max_score_final < 0.03:
            answerability = "LOW"
            refused = True
            reason = "Retrieved evidence was too weak to confidently answer."
        elif max_score_final < 0.07:
            answerability = "MEDIUM"
        else:
            answerability = "HIGH"

    # --- Generation (very lightweight in this demo) ---
    gen_start = time.perf_counter()
    if refused:
        answer = (
            "I cannot confidently answer this based on the CS 421 course PDFs in the "
            "RAG corpus. The retrieved evidence is either empty or too weak."
        )
    else:
        answer = _build_grounded_answer(req.question, citations)
    generation_ms = (time.perf_counter() - gen_start) * 1000.0

    total_ms = (time.perf_counter() - total_start) * 1000.0

    retrieval_graph = _build_retrieval_graph(req.question, citations)

    resp = RAGQueryResponse(
        answer=answer,
        answerability=answerability,
        refused=refused,
        reason=reason,
        citations=citations,
        retrieval_graph=retrieval_graph,
        timings_ms={
            "retrieval": round(retrieval_ms, 1),
            "generation": round(generation_ms, 1),
            "total": round(total_ms, 1),
        },
    )

    _log_query(
        req,
        resp,
        grounded=grounded and not refused,
        used_docs=list(used_docs),
    )
    return resp


def _build_grounded_answer(question: str, citations: List[Citation]) -> str:
    parts = [f"Q: {question}", "", "Grounded answer (built from course slides):"]

    for i, c in enumerate(citations[:3], start=1):
        parts.append(
            f"{i}. From {c.doc_title} (slide page {c.page_number}): "
            f"{c.snippet.strip()}..."
        )

    parts.append("")
    parts.append(
        "This answer is constructed only from the retrieved course material above."
    )
    return "\n".join(parts)


def _build_retrieval_graph(question: str, citations: List[Citation]) -> Dict:
    nodes = []
    edges = []

    # Query node
    nodes.append({"id": "query", "label": question, "type": "query"})

    docs_seen: Dict[str, str] = {}

    # Document nodes and edges query -> doc
    for c in citations:
        if c.doc_id not in docs_seen:
            doc_node_id = f"doc:{c.doc_id}"
            docs_seen[c.doc_id] = doc_node_id
            nodes.append({"id": doc_node_id, "label": c.doc_title, "type": "document"})
            edges.append(
                {
                    "source": "query",
                    "target": doc_node_id,
                    "weight": c.score_final,
                }
            )

    # Chunk nodes
    for idx, c in enumerate(citations):
        chunk_node_id = f"chunk:{idx}"
        nodes.append(
            {
                "id": chunk_node_id,
                "label": f"p{c.page_number}",
                "type": "chunk",
                "snippet": c.snippet,
                "glow": True,
            }
        )
        doc_node_id = docs_seen[c.doc_id]
        edges.append(
            {
                "source": doc_node_id,
                "target": chunk_node_id,
                "weight": c.score_final,
            }
        )

    return {"nodes": nodes, "edges": edges}


def _log_query(
    req: RAGQueryRequest,
    resp: RAGQueryResponse,
    grounded: bool,
    used_docs: List[str],
) -> None:
    log = QAlog(
        log_id=new_log_id(),
        timestamp=datetime.utcnow(),
        question=req.question,
        mode=req.mode,
        top_k=req.top_k,
        rerank=req.rerank,
        used_docs=used_docs,
        grounded=grounded,
        answerability=resp.answerability,
        refused=resp.refused,
        retrieval_ms=resp.timings_ms["retrieval"],
        generation_ms=resp.timings_ms["generation"],
        total_ms=resp.timings_ms["total"],
    )
    store.add_log(log)


# -------- DASHBOARD / OVERVIEW --------


def compute_overview_stats() -> Dict:
    num_docs = len(store.documents)
    num_chunks = len(store.chunks)
    num_questions = len(store.logs)

    if num_questions == 0:
        grounded_ratio = 0.0
    else:
        grounded_count = sum(1 for l in store.logs if l.grounded)
        grounded_ratio = grounded_count / num_questions

    mode_counts: Dict[str, int] = {}
    day_counts: Dict[str, int] = {}

    for l in store.logs:
        mode_counts[l.mode] = mode_counts.get(l.mode, 0) + 1
        day = l.timestamp.strftime("%a")  # Mon, Tue, ...
        day_counts[day] = day_counts.get(day, 0) + 1

    # Ensure days are in calendar order for chart
    ordered_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    questions_over_time = [
        {"day": d, "count": day_counts.get(d, 0)} for d in ordered_days
    ]

    return {
        "num_documents": num_docs,
        "num_chunks": num_chunks,
        "num_questions": num_questions,
        "grounded_ratio": grounded_ratio,
        "mode_counts": mode_counts,
        "questions_over_time": questions_over_time,
    }
