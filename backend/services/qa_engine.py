"""
Q&A Engine — orchestrates the full RAG answer cycle with caching and memory.
"""

import json
import uuid
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

from backend.utils.config import MAX_CONTEXT_WORDS, CACHE_TTL_HOURS, CACHE_SIMILARITY_THRESHOLD
from backend.embeddings.sentence_transformer import get_embedding
from backend.llm.flan_t5 import generate_text
from backend.memory.session_memory import SessionMemory
from backend.memory.long_term_memory import LongTermMemory
from backend.retrieval.retriever import expand_query, retrieve_chunks

logger = logging.getLogger(__name__)


def answer_question(
    query: str,
    collection,
    session: SessionMemory,
    long_term_memory: Optional[LongTermMemory] = None,
    interactions_col=None,
    cache_col=None,
) -> dict:
    """
    Full RAG answer cycle:
      1. Check qa_cache for near-identical recent query (cosine > 0.97, < 1hr).
      2. Expand query with session context.
      3. Retrieve relevant chunks.
      4. Pull top 3 LTM Q&A pairs, prepend to prompt.
      5. Cap context at 3000 words (trim oldest first).
      6. Generate answer via Gemini.
      7. Cache result and record interaction.

    Args:
        query: User's question.
        collection: ChromaDB document_chunks collection.
        session: Current session memory.
        long_term_memory: Optional LTM instance.
        interactions_col: Optional interaction history collection.
        cache_col: Optional QA cache collection.

    Returns:
        Dict with keys: answer (str), sources (list[dict]), query (str).
    """
    query_embedding = get_embedding(query, task_type="retrieval_query")

    # ── Cache check ──
    if cache_col is not None and cache_col.count() > 0:
        try:
            cr = cache_col.query(
                query_embeddings=[query_embedding], n_results=1,
                include=["documents", "metadatas", "distances"],
            )
            if cr["documents"][0]:
                similarity = max(0.0, 1.0 - cr["distances"][0][0])
                ct = datetime.fromisoformat(cr["metadatas"][0][0].get("cached_at", ""))
                if ct.tzinfo is None:
                    ct = ct.replace(tzinfo=timezone.utc)
                age_h = (datetime.now(timezone.utc) - ct).total_seconds() / 3600.0
                if similarity > CACHE_SIMILARITY_THRESHOLD and age_h < CACHE_TTL_HOURS:
                    logger.info("Returning cached answer")
                    return json.loads(cr["documents"][0][0])
        except Exception as e:
            logger.debug(f"Cache lookup failed: {e}")

    # ── Query expansion ──
    expanded = expand_query(query, session)

    # ── Retrieve chunks ──
    chunks = retrieve_chunks(expanded, collection)

    # ── LTM context ──
    ltm_context = ""
    if long_term_memory is not None:
        ltm_pairs = long_term_memory.retrieve_relevant(query, top_k=3)
        if ltm_pairs:
            lines = [f"Past Q: {p['query']}\nPast A: {p['answer']}" for p in ltm_pairs]
            ltm_context = "RELEVANT PAST Q&A:\n" + "\n\n".join(lines) + "\n\n"

    # ── Build context, cap at MAX_CONTEXT_WORDS ──
    budget = MAX_CONTEXT_WORDS - (len(ltm_context.split()) if ltm_context else 0)
    context_parts: list[str] = []
    sorted_chunks = sorted(chunks, key=lambda c: c["metadata"].get("ingested_at", ""), reverse=True)

    for chunk in sorted_chunks:
        cw = len(chunk["text"].split())
        if cw <= budget:
            context_parts.append(chunk["text"])
            budget -= cw
        else:
            trimmed = " ".join(chunk["text"].split()[:budget])
            if trimmed:
                context_parts.append(trimmed)
            break

    context = "\n\n".join(context_parts)

    # ── Generate answer ──
    prompt = (
        "You are a helpful assistant answering questions based on document excerpts. "
        "If the answer is not in the excerpts, say so clearly.\n\n"
        f"{ltm_context}DOCUMENT EXCERPTS:\n{context}\n\n"
        f"CONVERSATION HISTORY:\n{session.as_text()}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer concisely and cite which excerpt supports your answer."
    )

    try:
        answer = generate_text(prompt)
    except Exception as e:
        answer = f"Error generating answer: {e}"

    session.add("user", query)
    session.add("assistant", answer)

    result = {"answer": answer, "sources": chunks, "query": expanded}

    # ── Cache ──
    if cache_col is not None:
        try:
            cid = hashlib.md5(query.encode()).hexdigest()[:16]
            cache_col.upsert(
                ids=[cid], embeddings=[query_embedding],
                documents=[json.dumps(result, default=str)],
                metadatas=[{"cached_at": datetime.now(timezone.utc).isoformat()}],
            )
        except Exception as e:
            logger.debug(f"Cache store failed: {e}")

    # ── Record interaction ──
    if interactions_col is not None:
        try:
            iid = f"interaction_{uuid.uuid4().hex[:12]}"
            interactions_col.add(
                ids=[iid], embeddings=[query_embedding],
                documents=[json.dumps({
                    "query": query, "expanded_query": expanded, "answer": answer,
                    "chunks_used": [c["metadata"].get("source", "") for c in chunks],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })],
                metadatas=[{"query": query[:200], "timestamp": datetime.now(timezone.utc).isoformat()}],
            )
        except Exception as e:
            logger.debug(f"Interaction record failed: {e}")

    return result
