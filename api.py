"""
FastAPI server for the RAG Document Q&A System.

Endpoints:
  POST /upload    — Upload and ingest a document
  POST /ask       — Ask a question against ingested documents
  POST /feedback  — Submit feedback on an answer
  GET  /health    — Health check with collection stats
  GET  /memory    — Long-term memory statistics
  GET  /metrics   — System-wide metrics dashboard data
"""

import os
import tempfile
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.models.schemas import (
    AskRequest, AskResponse,
    FeedbackRequest, FeedbackResponse,
    HealthResponse, MemoryStatsResponse, MetricsResponse,
)
from backend.retrieval.vector_db import init_vector_db
from backend.services.document_processor import ingest_document
from backend.services.qa_engine import answer_question
from backend.memory.session_memory import SessionMemory
from backend.memory.long_term_memory import LongTermMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── App setup ───
app = FastAPI(
    title="RAG Document Q&A API",
    description="Intelligent Document Q&A with Long-term Memory — Google Gemini",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Shared state ───
_, chunks_col, interactions_col, cache_col = init_vector_db()
ltm = LongTermMemory()
sessions: dict[str, SessionMemory] = {}


def _get_session(session_id: str) -> SessionMemory:
    if session_id not in sessions:
        sessions[session_id] = SessionMemory()
    return sessions[session_id]


# ─── Endpoints ───

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """
    Upload and ingest a document. Accepts PDF, DOCX, HTML, TXT, MD.

    Returns:
        Dict with doc_id, filename, and status.
    """
    allowed = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {', '.join(allowed)}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        doc_id = ingest_document(tmp_path, chunks_col)
        return {"doc_id": doc_id, "filename": file.filename, "status": "ingested"}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(500, f"Ingestion failed: {str(e)}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> dict:
    """
    Ask a question against ingested documents.

    Returns:
        Dict with answer, sources, and expanded query.
    """
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    session = _get_session(request.session_id)

    try:
        result = answer_question(
            query=request.query,
            collection=chunks_col,
            session=session,
            long_term_memory=ltm,
            interactions_col=interactions_col,
            cache_col=cache_col,
        )
        sources = [
            {
                "source": s["metadata"].get("source", "unknown"),
                "chunk_index": s["metadata"].get("chunk_index", -1),
                "score": round(s["score"], 4),
                "page_number": s["metadata"].get("page_number", -1),
            }
            for s in result.get("sources", [])[:5]
        ]
        return {"answer": result["answer"], "sources": sources, "query": result["query"]}
    except Exception as e:
        logger.error(f"Q&A failed: {e}")
        raise HTTPException(500, f"Q&A failed: {str(e)}")


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> dict:
    """
    Submit user feedback (1-5 rating with optional correction).

    Returns:
        Confirmation dict.
    """
    try:
        ltm.record_feedback(
            query=request.query, answer=request.answer,
            rating=request.rating, correction=request.correction,
        )
        return {
            "status": "feedback_recorded",
            "rating": request.rating,
            "correction_stored": bool(request.correction and request.rating <= 2),
        }
    except Exception as e:
        raise HTTPException(500, f"Feedback recording failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health() -> dict:
    """Health check with collection counts."""
    try:
        return {
            "status": "ok",
            "collections": {
                "chunks": chunks_col.count(),
                "ltm": ltm.ltm_collection.count(),
                "interactions": interactions_col.count(),
                "cache": cache_col.count(),
            },
        }
    except Exception as e:
        return {"status": "degraded", "collections": {"error": str(e)}}


@app.get("/memory", response_model=MemoryStatsResponse)
async def memory_stats() -> dict:
    """Long-term memory statistics."""
    return ltm.get_stats()


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> dict:
    """System-wide metrics for the admin dashboard."""
    return {
        "total_documents": chunks_col.count(),
        "total_interactions": interactions_col.count(),
        "cached_queries": cache_col.count(),
        "memory_stats": ltm.get_stats(),
    }


# ─── Run ───
if __name__ == "__main__":
    import uvicorn
    from backend.utils.config import API_HOST, API_PORT
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)

