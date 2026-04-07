"""
Retrieval engine — query expansion, hybrid retrieval, and re-ranking.
"""

import logging
from datetime import datetime, timezone

from backend.utils.config import TOP_K_RETRIEVAL, RELEVANCE_WEIGHT, RECENCY_WEIGHT
from backend.embeddings.sentence_transformer import get_embedding
from backend.llm.flan_t5 import generate_text
from backend.memory.session_memory import SessionMemory

logger = logging.getLogger(__name__)


def expand_query(raw_query: str, session: SessionMemory) -> str:
    """
    Rewrite a follow-up query into a self-contained question using session context.

    Args:
        raw_query: The user's original query.
        session: Current SessionMemory with conversation history.

    Returns:
        A self-contained rewritten query string.
    """
    if not session.exchanges:
        return raw_query

    history = session.recent_as_text(n=6)
    prompt = (
        "Rewrite the user's question to be self-contained using the history.\n\n"
        f"HISTORY:\n{history}\n\n"
        f"QUESTION: {raw_query}\n\n"
        "REWRITTEN QUESTION:"
    )
    try:
        rewritten = generate_text(prompt, max_length=100)
        return rewritten if rewritten else raw_query
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return raw_query


def retrieve_chunks(
    query: str,
    collection,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    """
    Retrieve and re-rank document chunks for a query.

    Re-ranking formula: score = 0.7 * cosine_similarity + 0.3 * recency_score
    where recency_score = 1 / (1 + days_since_ingestion).

    Args:
        query: Search query string.
        collection: ChromaDB collection.
        top_k: Number of results after re-ranking.

    Returns:
        List of dicts with keys: text, metadata, score.
    """
    query_embedding = get_embedding(query, task_type="retrieval_query")

    fetch_k = min(top_k * 2, max(collection.count(), 1))
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    now = datetime.now(timezone.utc)
    chunks: list[dict] = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if meta.get("type") == "doc_summary":
            continue

        cosine_score = max(0.0, 1.0 - dist)

        try:
            ingested = datetime.fromisoformat(meta.get("ingested_at", now.isoformat()))
            if ingested.tzinfo is None:
                ingested = ingested.replace(tzinfo=timezone.utc)
            days = (now - ingested).total_seconds() / 86400.0
        except (ValueError, TypeError):
            days = 0.0

        recency = 1.0 / (1.0 + days)
        combined = RELEVANCE_WEIGHT * cosine_score + RECENCY_WEIGHT * recency

        chunks.append({"text": doc, "metadata": meta, "score": combined})

    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks[:top_k]
