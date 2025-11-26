from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from .storage import store
from .schemas import (
    DocumentOut,
    DocumentListOut,
    DocUploadResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    LogsResponse,
    LogItem,
    ConceptGraphResponse,
    ConceptNodeSchema,
    ConceptEdgeSchema,
    OverviewStats,
)
from .rag_engine import ingest_pdf, query_rag, compute_overview_stats
from .concept_graph import build_concept_graph


app = FastAPI(title="RAGCourseLab API")

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------- DOCS ----------


@app.post("/api/docs", response_model=DocUploadResponse)
async def upload_doc(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    doc = ingest_pdf(file_bytes, file.filename, source_type="slides", topics=[])

    return DocUploadResponse(
        doc_id=doc.doc_id,
        title=doc.title,
        num_chunks=doc.num_chunks,
    )


@app.get("/api/docs", response_model=DocumentListOut)
async def list_docs():
    docs_out: List[DocumentOut] = []
    for d in store.documents.values():
        docs_out.append(
            DocumentOut(
                doc_id=d.doc_id,
                title=d.title,
                source_type=d.source_type,
                topics=d.topics,
                num_chunks=d.num_chunks,
                created_at=d.created_at.isoformat(),
                updated_at=d.updated_at.isoformat(),
            )
        )
    return DocumentListOut(documents=docs_out)


# ---------- RAG QUERY ----------


@app.post("/api/rag/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    return query_rag(req)


# ---------- LOGS / OVERVIEW ----------


@app.get("/api/rag/logs", response_model=LogsResponse)
async def get_logs():
    items: List[LogItem] = []
    for l in sorted(store.logs, key=lambda x: x.timestamp, reverse=True):
        items.append(
            LogItem(
                log_id=l.log_id,
                timestamp=l.timestamp.isoformat(),
                question=l.question,
                mode=l.mode,
                used_docs=l.used_docs,
                grounded=l.grounded,
                answerability=l.answerability,
            )
        )
    return LogsResponse(logs=items)


@app.get("/api/rag/overview", response_model=OverviewStats)
async def get_overview():
    stats = compute_overview_stats()
    return OverviewStats(**stats)


# ---------- CONCEPT GRAPH ----------


@app.get("/api/concept-graph", response_model=ConceptGraphResponse)
async def get_concept_graph():
    graph = build_concept_graph()
    nodes = [
        ConceptNodeSchema(id=n["id"], label=n["label"], type=n["type"])
        for n in graph["nodes"]
    ]
    edges = [
        ConceptEdgeSchema(
            source=e["source"], target=e["target"], weight=float(e["weight"])
        )
        for e in graph["edges"]
    ]
    return ConceptGraphResponse(nodes=nodes, edges=edges)
