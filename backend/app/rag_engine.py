from __future__ import annotations
from typing import List, Tuple, Dict
from uuid import uuid4
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PyPDF2 import PdfReader

from .storage import store
from .models import Document, Chunk, QAlog
from .schemas import Citation, RAGQueryRequest, RAGQueryResponse


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    source_type: str = "slides",
    topics: List[str] | None = None,
) -> Document:
    topics = topics or []
    doc_id = filename.split(".pdf")[0].replace(" ", "_").lower()
    if doc_id in store.documents:
        # simple version: append more chunks to an existing doc id
        doc = store.documents[doc_id]
    else:
        doc = Document(
            doc_id=doc_id,
            title=filename,
            source_type=source_type,
            topics=topics,
        )
        store.add_document(doc)

    reader = PdfReader(io_bytes(file_bytes))
    existing_chunks = len(store.chunks_by_doc.get(doc.doc_id, []))
    chunk_index = existing_chunks

    for page_idx, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        # simple page-level chunk
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


def io_bytes(data: bytes):
    from io import BytesIO
    return BytesIO(data)


def rebuild_index() -> None:
    texts = []
    chunk_ids = []

    for chunk_id, chunk in store.chunks.items():
        texts.append(chunk.text)
        chunk_ids.append(chunk_id)

    if not texts:
        store.vectorizer = None
        store.tfidf_matrix = None
        store.chunk_id_to_index = {}
        return

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)

    store.vectorizer = vectorizer
    store.tfidf_matrix = tfidf_matrix
    store.chunk_id_to_index = {cid: idx for idx, cid in enumerate(chunk_ids)}


def query_rag(req: RAGQueryRequest) -> RAGQueryResponse:
    # If no documents, answer with low confidence
    if store.vectorizer is None or store.tfidf_matrix is None:
        empty_resp = RAGQueryResponse(
            answer="No documents have been ingested yet. Please upload course PDFs first.",
            citations=[],
            answerability="LOW",
            retrieval_graph={"nodes": [], "edges": []},
        )
        log_query(req, empty_resp, grounded=False, used_docs=[])
        return empty_resp

    query_vec = store.vectorizer.transform([req.question])
    cosine_similarities = linear_kernel(query_vec, store.tfidf_matrix).flatten()

    top_k = min(req.top_k, len(cosine_similarities))
    best_indices = np.argsort(-cosine_similarities)[:top_k]

    citations: List[Citation] = []
    used_docs: set[str] = set()

    for idx in best_indices:
        score = float(cosine_similarities[idx])
        if score <= 0:
            continue
        chunk_id = list(store.chunk_id_to_index.keys())[idx]
        chunk = store.chunks[chunk_id]
        doc = store.documents[chunk.doc_id]
        snippet = chunk.text[:300].replace("\n", " ")

        # we fake "bm25" vs "dense" scores, but keep them proportional
        score_bm25 = score
        score_dense = score * 0.9 if req.mode != "bm25" else score * 0.8
        score_final = (score_bm25 + score_dense) / 2.0

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
    answerability = "HIGH" if grounded else "LOW"

    if grounded:
        # naive "answer": combine snippets
        answer = build_grounded_answer(req.question, citations)
    else:
        answer = (
            "I could not find any strong supporting evidence in the ingested course PDFs "
            "for this question, so I prefer not to answer."
        )

    retrieval_graph = build_retrieval_graph(req.question, citations)

    resp = RAGQueryResponse(
        answer=answer,
        citations=citations,
        answerability=answerability,
        retrieval_graph=retrieval_graph,
    )

    log_query(req, resp, grounded=grounded, used_docs=list(used_docs))
    return resp


def build_grounded_answer(question: str, citations: List[Citation]) -> str:
    parts = [f"Q: {question}", "", "Grounded answer (built from course slides):"]
    # use at most 3 citations to keep it short-ish
    for i, c in enumerate(citations[:3], start=1):
        parts.append(
            f"{i}. From {c.doc_title} (slide page {c.page_number}): {c.snippet.strip()}..."
        )
    parts.append("")
    parts.append(
        "This answer is constructed only from the retrieved course material above."
    )
    return "\n".join(parts)


def build_retrieval_graph(question: str, citations: List[Citation]) -> Dict:
    nodes = []
    edges = []

    # Query node
    nodes.append({"id": "query", "label": question, "type": "query"})

    # Document nodes
    docs_seen: Dict[str, str] = {}
    for c in citations:
        if c.doc_id not in docs_seen:
            doc_node_id = f"doc:{c.doc_id}"
            docs_seen[c.doc_id] = doc_node_id
            nodes.append({"id": doc_node_id, "label": c.doc_title, "type": "document"})
            # edge from query -> doc with weight from best citation score
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


def log_query(
    req: RAGQueryRequest,
    resp: RAGQueryResponse,
    grounded: bool,
    used_docs: List[str],
) -> None:
    log = QAlog(
        log_id=store.next_log_id(),
        timestamp=datetime.utcnow(),
        question=req.question,
        mode=req.mode,
        used_docs=used_docs,
        grounded=grounded,
        answerability=resp.answerability,
    )
    store.add_log(log)


def compute_overview_stats():
    num_docs = len(store.documents)
    num_chunks = len(store.chunks)
    num_questions = len(store.logs)

    if num_questions == 0:
        grounded_ratio = 0.0
    else:
        grounded_count = sum(1 for l in store.logs if l.grounded)
        grounded_ratio = grounded_count / num_questions

    mode_counts: Dict[str, int] = {}
    for l in store.logs:
        mode_counts[l.mode] = mode_counts.get(l.mode, 0) + 1

    return {
        "num_documents": num_docs,
        "num_chunks": num_chunks,
        "num_questions": num_questions,
        "grounded_ratio": grounded_ratio,
        "mode_counts": mode_counts,
    }
