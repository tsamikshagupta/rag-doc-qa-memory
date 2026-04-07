"""
Exercise 1: Intelligent Document Q&A System with Memory
Focus: RAG Architecture with Long-term Memory using Google Gemini
"""

# ============================================================
# DEPENDENCIES — run before starting:
#   pip install google-generativeai chromadb pypdf python-docx
#          beautifulsoup4 python-dotenv fastapi uvicorn
#          python-multipart numpy requests
# ============================================================

import os
import uuid
import json
import hashlib
import time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb.config import Settings

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
CHROMA_PATH = "./chroma_store"
CHUNK_SIZE = 512          # words per chunk
CHUNK_OVERLAP = 64        # overlap between adjacent chunks (words)
SHORT_TERM_LIMIT = 20     # max exchanges kept in session memory
TOP_K_RETRIEVAL = 5       # chunks to retrieve per query

genai.configure(api_key=GEMINI_API_KEY)


# ─────────────────────────────────────────────
# GEMINI WRAPPER UTILITIES
# ─────────────────────────────────────────────

def call_gemini_pro(prompt: str, max_retries: int = 1) -> str:
    """
    Call Gemini Pro (gemini-1.5-flash) with retry on quota errors.

    Args:
        prompt: The text prompt to send.
        max_retries: Number of retries on quota/rate-limit errors.

    Returns:
        The generated text response as a string.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "rate" in error_str or "429" in error_str:
                if attempt < max_retries:
                    print(f"[gemini] Quota/rate limit hit, waiting 60s before retry...")
                    time.sleep(60)
                    continue
            raise


# ─────────────────────────────────────────────
# PHASE 1 — DOCUMENT PROCESSING PIPELINE
# ─────────────────────────────────────────────

def load_document(file_path: str) -> str:
    """
    Load and extract raw text from a document file.

    Supported formats:
      - PDF  (.pdf)  — uses pypdf PdfReader, extracts text per page
      - DOCX (.docx) — uses python-docx Document, iterates paragraphs
      - HTML (.html/.htm) — uses BeautifulSoup to strip tags and extract text
      - TXT  (.txt)  — plain file read
      - MD   (.md)   — plain file read

    Args:
        file_path: Absolute or relative path to the document file.

    Returns:
        The full extracted text as a single string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages_text: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n\n".join(pages_text)

    elif ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        paragraphs: list[str] = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        return "\n\n".join(paragraphs)

    elif ext in (".html", ".htm"):
        from bs4 import BeautifulSoup
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into overlapping chunks using a two-pass strategy:
      1. Primary: split on double-newline (paragraph boundaries).
         Adjacent short paragraphs are merged up to chunk_size words.
      2. Fallback: any paragraph block exceeding chunk_size words is
         subdivided with a sliding window of chunk_size words and
         CHUNK_OVERLAP overlap.

    Each returned chunk dict contains:
      - text (str):        the chunk content
      - chunk_index (int): sequential index
      - char_start (int):  character offset in the original text
      - char_end (int):    character end offset in the original text

    Args:
        text: The full document text.
        chunk_size: Maximum words per chunk.
        overlap: Word overlap between sliding-window sub-chunks.

    Returns:
        A list of chunk dictionaries.
    """
    # ── Step 1: paragraph-level split ──
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # ── Step 2: merge short paragraphs, split long ones ──
    raw_blocks: list[str] = []
    current_block: list[str] = []
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_word_count + para_words <= chunk_size:
            current_block.append(para)
            current_word_count += para_words
        else:
            # flush current block
            if current_block:
                raw_blocks.append("\n\n".join(current_block))
            # if the single paragraph itself is too large, keep it for sliding-window
            current_block = [para]
            current_word_count = para_words

    if current_block:
        raw_blocks.append("\n\n".join(current_block))

    # ── Step 3: sliding-window fallback for oversized blocks ──
    final_texts: list[str] = []
    for block in raw_blocks:
        words = block.split()
        if len(words) <= chunk_size:
            final_texts.append(block)
        else:
            step = max(1, chunk_size - overlap)
            for start in range(0, len(words), step):
                window = words[start : start + chunk_size]
                final_texts.append(" ".join(window))
                if start + chunk_size >= len(words):
                    break

    # ── Step 4: build chunk dicts with char offsets ──
    chunks: list[dict] = []
    search_start = 0
    for idx, chunk_text_str in enumerate(final_texts):
        # locate the chunk in the original text for char offsets
        # use first 80 chars as anchor for fuzzy location
        anchor = chunk_text_str[:80]
        char_start = text.find(anchor, search_start)
        if char_start == -1:
            char_start = search_start  # fallback
        char_end = char_start + len(chunk_text_str)

        chunks.append({
            "text": chunk_text_str,
            "chunk_index": idx,
            "char_start": char_start,
            "char_end": char_end,
        })
        search_start = max(search_start, char_start + 1)

    return chunks


def get_embedding(text: str, task_type: str = "retrieval_document") -> list[float]:
    """
    Generate an embedding vector using Gemini's embedding model.

    Wraps genai.embed_content with retry logic for quota errors.

    Args:
        text: The text to embed.
        task_type: One of 'retrieval_document', 'retrieval_query', etc.

    Returns:
        A list of floats representing the embedding vector.
    """
    for attempt in range(2):
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type=task_type,
            )
            return result["embedding"]
        except Exception as e:
            error_str = str(e).lower()
            if ("quota" in error_str or "rate" in error_str or "429" in error_str) and attempt == 0:
                print("[embedding] Quota/rate limit hit, waiting 60s...")
                time.sleep(60)
                continue
            raise


def ingest_document(file_path: str, collection) -> str:
    """
    Full ingestion pipeline: load → chunk → embed → store in ChromaDB.

    In addition to per-chunk embeddings, generates ONE document-level
    embedding (full text truncated to 2000 words) stored with
    metadata type='doc_summary'.

    Metadata stored per chunk:
      - source: original file path
      - doc_id: deterministic hash of file path
      - chunk_index: sequential chunk number
      - page_number: (PDF only) approximate page number
      - ingested_at: UTC ISO timestamp
      - char_start / char_end: character offsets in original text

    Args:
        file_path: Path to the document to ingest.
        collection: ChromaDB collection to store chunks in.

    Returns:
        The generated doc_id string.
    """
    doc_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
    raw_text = load_document(file_path)
    chunks = chunk_text(raw_text)
    ext = Path(file_path).suffix.lower()
    now = datetime.now(timezone.utc).isoformat()

    # ── Per-chunk embeddings ──
    for chunk in chunks:
        embedding = get_embedding(chunk["text"])
        chunk_id = f"{doc_id}_chunk_{chunk['chunk_index']}"

        # Approximate page number for PDFs (rough heuristic: ~3000 chars/page)
        page_number = -1
        if ext == ".pdf":
            page_number = chunk["char_start"] // 3000 + 1

        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "doc_id": doc_id,
                "source": file_path,
                "chunk_index": chunk["chunk_index"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "page_number": page_number,
                "ingested_at": now,
                "type": "chunk",
            }],
        )

    # ── Document-level summary embedding ──
    summary_words = raw_text.split()[:2000]
    summary_text = " ".join(summary_words)
    summary_embedding = get_embedding(summary_text)
    summary_id = f"{doc_id}_summary"

    collection.add(
        ids=[summary_id],
        embeddings=[summary_embedding],
        documents=[summary_text],
        metadatas=[{
            "doc_id": doc_id,
            "source": file_path,
            "chunk_index": -1,
            "char_start": 0,
            "char_end": len(summary_text),
            "page_number": -1,
            "ingested_at": now,
            "type": "doc_summary",
        }],
    )

    print(f"[ingest] {file_path} → {len(chunks)} chunks + 1 summary stored (doc_id={doc_id})")
    return doc_id


# ─────────────────────────────────────────────
# PHASE 2 — Q&A ENGINE WITH MEMORY
# ─────────────────────────────────────────────

class SessionMemory:
    """Short-term memory: last N conversation exchanges."""

    def __init__(self, limit: int = SHORT_TERM_LIMIT):
        self.exchanges: list[dict] = []
        self.limit = limit

    def add(self, role: str, content: str) -> None:
        """Append a role/content pair and trim to the limit."""
        self.exchanges.append({"role": role, "content": content})
        if len(self.exchanges) > self.limit * 2:
            self.exchanges = self.exchanges[-self.limit * 2 :]

    def recent(self, n: int = 6) -> list[dict]:
        """Return the last n exchanges."""
        return self.exchanges[-(n * 2):]

    def as_text(self) -> str:
        """Format all exchanges as a readable transcript."""
        return "\n".join(f"{e['role'].upper()}: {e['content']}" for e in self.exchanges)

    def recent_as_text(self, n: int = 6) -> str:
        """Format the last n exchanges as text."""
        return "\n".join(
            f"{e['role'].upper()}: {e['content']}" for e in self.recent(n)
        )


def expand_query(raw_query: str, session: SessionMemory) -> str:
    """
    Use Gemini to rewrite a follow-up query into a self-contained question
    by resolving pronouns and references using the last 6 session exchanges.

    If the session is empty, returns the raw query unchanged.

    Args:
        raw_query: The user's original query text.
        session: The current SessionMemory instance.

    Returns:
        A self-contained rewritten query string.
    """
    if not session.exchanges:
        return raw_query

    history = session.recent_as_text(n=6)
    prompt = f"""You are a query rewriting assistant. Your ONLY job is to rewrite
the user's new question so it is completely self-contained — resolve any pronouns,
references, or ellipsis using the conversation history below.

CONVERSATION HISTORY (last 6 exchanges):
{history}

NEW QUESTION: {raw_query}

Return ONLY the rewritten question. Do not answer it. Do not add commentary."""

    try:
        rewritten = call_gemini_pro(prompt)
        return rewritten if rewritten else raw_query
    except Exception as e:
        print(f"[expand_query] Gemini call failed ({e}), using raw query")
        return raw_query


def retrieve_chunks(
    query: str,
    collection,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    """
    Retrieve and re-rank document chunks for a query.

    Process:
      1. Embed the query with task_type='retrieval_query'.
      2. Fetch top_k * 2 candidates from ChromaDB (over-fetch for re-ranking).
      3. Re-rank using: weight = 0.7 * cosine_score + 0.3 * recency_score
         where recency_score = 1 / (1 + days_since_ingestion).
      4. Filter out doc_summary entries, return top_k results.

    Args:
        query: The search query string.
        collection: The ChromaDB collection to search.
        top_k: Number of results to return after re-ranking.

    Returns:
        A list of dicts with keys: text, metadata, score.
    """
    query_embedding = get_embedding(query, task_type="retrieval_query")

    # Over-fetch to allow re-ranking
    fetch_k = min(top_k * 2, collection.count() or top_k)
    if fetch_k == 0:
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
        # Skip document-level summaries from retrieval results
        if meta.get("type") == "doc_summary":
            continue

        # ChromaDB L2 distance → cosine similarity approximation
        cosine_score = max(0.0, 1.0 - dist)

        # Recency score
        try:
            ingested = datetime.fromisoformat(meta.get("ingested_at", now.isoformat()))
            if ingested.tzinfo is None:
                ingested = ingested.replace(tzinfo=timezone.utc)
            days_since = (now - ingested).total_seconds() / 86400.0
        except (ValueError, TypeError):
            days_since = 0.0

        recency_score = 1.0 / (1.0 + days_since)

        combined_score = 0.7 * cosine_score + 0.3 * recency_score

        chunks.append({
            "text": doc,
            "metadata": meta,
            "score": combined_score,
        })

    # Sort by combined score descending, return top_k
    chunks.sort(key=lambda c: c["score"], reverse=True)
    return chunks[:top_k]


def answer_question(
    query: str,
    collection,
    session: SessionMemory,
    long_term_memory: Optional["LongTermMemory"] = None,
    interactions_col=None,
    cache_col=None,
) -> dict:
    """
    Full RAG answer cycle:
      1. Check qa_cache for near-identical recent query (cosine > 0.97, < 1hr old).
      2. Expand query using session context.
      3. Retrieve relevant chunks.
      4. Pull top 3 long-term memory Q&A pairs and prepend to prompt.
      5. Cap context at 3000 words; trim oldest chunks first if over.
      6. Generate answer via Gemini.
      7. Cache the result and record in interaction_history.

    Args:
        query: The user's question.
        collection: ChromaDB collection of document chunks.
        session: Current session memory.
        long_term_memory: Optional LongTermMemory instance.
        interactions_col: Optional ChromaDB collection for interaction history.
        cache_col: Optional ChromaDB collection for QA cache.

    Returns:
        Dict with keys: answer (str), sources (list[dict]), query (str).
    """
    query_embedding = get_embedding(query, task_type="retrieval_query")

    # ── Cache check ──
    if cache_col is not None and cache_col.count() > 0:
        try:
            cache_results = cache_col.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )
            if cache_results["documents"][0]:
                dist = cache_results["distances"][0][0]
                similarity = max(0.0, 1.0 - dist)
                cached_meta = cache_results["metadatas"][0][0]
                cached_time = datetime.fromisoformat(cached_meta.get("cached_at", ""))
                if cached_time.tzinfo is None:
                    cached_time = cached_time.replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - cached_time).total_seconds() / 3600.0

                if similarity > 0.97 and age_hours < 1.0:
                    cached_data = json.loads(cache_results["documents"][0][0])
                    print("[cache] Returning cached answer")
                    return cached_data
        except Exception as e:
            print(f"[cache] Cache lookup failed ({e}), proceeding with generation")

    # ── Query expansion ──
    expanded = expand_query(query, session)

    # ── Retrieve chunks ──
    chunks = retrieve_chunks(expanded, collection)

    # ── Long-term memory context ──
    ltm_context = ""
    if long_term_memory is not None:
        ltm_pairs = long_term_memory.retrieve_relevant(query, top_k=3)
        if ltm_pairs:
            ltm_lines = []
            for pair in ltm_pairs:
                ltm_lines.append(f"Past Q: {pair['query']}\nPast A: {pair['answer']}")
            ltm_context = "RELEVANT PAST Q&A (for reference):\n" + "\n\n".join(ltm_lines) + "\n\n"

    # ── Build context, cap at 3000 words ──
    max_context_words = 3000
    context_parts: list[str] = []
    word_budget = max_context_words

    # Count LTM words first
    ltm_words = len(ltm_context.split()) if ltm_context else 0
    word_budget -= ltm_words

    # Add chunks newest-first (so oldest get trimmed first)
    sorted_by_recency = sorted(
        chunks,
        key=lambda c: c["metadata"].get("ingested_at", ""),
        reverse=True,
    )

    for chunk in sorted_by_recency:
        chunk_words = len(chunk["text"].split())
        if chunk_words <= word_budget:
            context_parts.append(chunk["text"])
            word_budget -= chunk_words
        else:
            # Trim to fit remaining budget
            trimmed = " ".join(chunk["text"].split()[:word_budget])
            if trimmed:
                context_parts.append(trimmed)
            break

    context = "\n\n".join(context_parts)

    # ── Generate answer ──
    prompt = f"""You are a helpful assistant answering questions based on the
provided document excerpts. If the answer is not in the excerpts, say so clearly.

{ltm_context}DOCUMENT EXCERPTS:
{context}

CONVERSATION HISTORY:
{session.as_text()}

QUESTION: {query}

Answer concisely and cite which excerpt supports your answer."""

    try:
        answer = call_gemini_pro(prompt)
    except Exception as e:
        answer = f"Error generating answer: {e}"

    # ── Update session ──
    session.add("user", query)
    session.add("assistant", answer)

    result = {"answer": answer, "sources": chunks, "query": expanded}

    # ── Cache the result ──
    if cache_col is not None:
        try:
            cache_id = hashlib.md5(query.encode()).hexdigest()[:16]
            cache_col.upsert(
                ids=[cache_id],
                embeddings=[query_embedding],
                documents=[json.dumps(result, default=str)],
                metadatas=[{"cached_at": datetime.now(timezone.utc).isoformat()}],
            )
        except Exception as e:
            print(f"[cache] Failed to cache result: {e}")

    # ── Record interaction ──
    if interactions_col is not None:
        try:
            interaction_id = f"interaction_{uuid.uuid4().hex[:12]}"
            interactions_col.add(
                ids=[interaction_id],
                embeddings=[query_embedding],
                documents=[json.dumps({
                    "query": query,
                    "expanded_query": expanded,
                    "answer": answer,
                    "chunks_used": [c["metadata"].get("source", "") for c in chunks],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })],
                metadatas=[{
                    "query": query[:200],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
            )
        except Exception as e:
            print(f"[interactions] Failed to record interaction: {e}")

    return result


# ─────────────────────────────────────────────
# PHASE 3 — LEARNING & ADAPTATION
# ─────────────────────────────────────────────

class LongTermMemory:
    """
    Persists successful Q&A pairs, user corrections, and interaction patterns.

    Features:
      - Stores Q&A pairs with Gemini-classified topic labels
      - Maintains a ChromaDB collection (ltm_embeddings) for semantic retrieval
      - Tracks corrections for learning from user feedback
      - Filters by rating >= 3 during retrieval

    Args:
        store_path: Path to the JSON persistence file.
        chroma_path: Path for ChromaDB persistent storage.
    """

    def __init__(
        self,
        store_path: str = "./long_term_memory.json",
        chroma_path: str = CHROMA_PATH,
    ):
        self.store_path = store_path
        self.data: dict = {
            "qa_pairs": [],
            "corrections": [],
            "interaction_history": [],
        }
        self._load()

        # LTM embeddings collection in ChromaDB
        client = chromadb.PersistentClient(path=chroma_path)
        self.ltm_collection = client.get_or_create_collection("ltm_embeddings")

    def _load(self) -> None:
        """Load persisted data from JSON file if it exists."""
        if Path(self.store_path).exists():
            with open(self.store_path) as f:
                self.data = json.load(f)

    def _save(self) -> None:
        """Persist current data to the JSON file."""
        with open(self.store_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def record_feedback(
        self,
        query: str,
        answer: str,
        rating: int,
        correction: Optional[str] = None,
    ) -> None:
        """
        Record user feedback on a Q&A interaction.

        After saving, calls Gemini to classify the query topic into one of:
        legal, medical, finance, technical, general. The topic is stored
        alongside the Q&A pair.

        If rating <= 2 and a correction is provided, it is stored in the
        corrections list for future learning.

        The Q&A pair embedding is also stored in the ltm_embeddings ChromaDB
        collection for semantic retrieval.

        Args:
            query: The original user query.
            answer: The system's answer.
            rating: User rating from 1-5.
            correction: Optional better answer provided by the user.
        """
        # ── Classify topic with Gemini ──
        topic = "general"
        try:
            classification_prompt = f"""Classify the following query into exactly ONE of these topics:
legal, medical, finance, technical, general

Query: {query}

Return ONLY the single topic word, nothing else."""
            topic = call_gemini_pro(classification_prompt).strip().lower()
            if topic not in ("legal", "medical", "finance", "technical", "general"):
                topic = "general"
        except Exception as e:
            print(f"[ltm] Topic classification failed ({e}), defaulting to 'general'")

        entry = {
            "query": query,
            "answer": answer,
            "rating": rating,
            "topic": topic,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.data["qa_pairs"].append(entry)

        if correction and rating <= 2:
            self.data["corrections"].append({
                "query": query,
                "original_answer": answer,
                "correction": correction,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        self._save()

        # ── Store embedding in ChromaDB for semantic retrieval ──
        try:
            qa_text = f"Q: {query}\nA: {answer}"
            embedding = get_embedding(qa_text)
            ltm_id = f"ltm_{hashlib.md5(qa_text.encode()).hexdigest()[:12]}"
            self.ltm_collection.upsert(
                ids=[ltm_id],
                embeddings=[embedding],
                documents=[json.dumps(entry)],
                metadatas=[{
                    "query": query[:200],
                    "rating": rating,
                    "topic": topic,
                    "timestamp": entry["timestamp"],
                }],
            )
        except Exception as e:
            print(f"[ltm] Failed to store LTM embedding: {e}")

    def retrieve_relevant(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve the most semantically similar past Q&A pairs.

        Embeds the query and searches the ltm_embeddings ChromaDB collection.
        Only returns pairs with rating >= 3.

        Falls back to returning the most recent highly-rated pairs from the
        JSON store if the ChromaDB lookup fails.

        Args:
            query: The current user query.
            top_k: Maximum number of pairs to return.

        Returns:
            A list of Q&A pair dicts (with keys: query, answer, rating, topic, timestamp).
        """
        # Try semantic retrieval from ChromaDB
        if self.ltm_collection.count() > 0:
            try:
                query_embedding = get_embedding(query, task_type="retrieval_query")
                results = self.ltm_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k * 2,  # over-fetch to filter
                    include=["documents", "metadatas", "distances"],
                    where={"rating": {"$gte": 3}},
                )

                pairs: list[dict] = []
                for doc_str, meta in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                ):
                    try:
                        pair = json.loads(doc_str)
                        pairs.append(pair)
                    except json.JSONDecodeError:
                        continue

                return pairs[:top_k]
            except Exception as e:
                print(f"[ltm] Semantic retrieval failed ({e}), using fallback")

        # Fallback: most recent highly-rated pairs
        high_rated = [p for p in self.data["qa_pairs"] if p.get("rating", 0) >= 3]
        return high_rated[-top_k:]


def collect_feedback(query: str, answer: str, ltm: LongTermMemory) -> None:
    """
    CLI helper to collect explicit feedback after an answer.
    Replace with a proper UI widget in production.

    Args:
        query: The original user query.
        answer: The system's generated answer.
        ltm: The LongTermMemory instance to record feedback in.
    """
    try:
        rating_str = input("Rate this answer 1–5 (press Enter to skip): ").strip()
        rating = int(rating_str) if rating_str else 0
    except ValueError:
        rating = 0

    correction = None
    if rating and rating <= 2:
        correction = input("What's a better answer? (press Enter to skip): ").strip() or None

    if rating:
        ltm.record_feedback(query, answer, rating, correction)
        print("[memory] Feedback saved.")


# ─────────────────────────────────────────────
# PHASE 4 — EVALUATION HELPERS
# ─────────────────────────────────────────────

def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 score (SQuAD-style).

    Handles unanswerable questions: if both prediction and ground truth
    indicate no answer, F1 = 1.0. If only one does, F1 = 0.0.

    Args:
        prediction: The predicted answer string.
        ground_truth: The gold-standard answer string.

    Returns:
        F1 score as a float between 0.0 and 1.0.
    """
    # Handle unanswerable
    pred_lower = prediction.strip().lower()
    gt_lower = ground_truth.strip().lower()

    no_answer_signals = ["unanswerable", "no answer", "cannot be answered", "not in the"]

    pred_unanswerable = any(s in pred_lower for s in no_answer_signals) or not pred_lower
    gt_unanswerable = not gt_lower

    if pred_unanswerable and gt_unanswerable:
        return 1.0
    if pred_unanswerable or gt_unanswerable:
        return 0.0

    pred_tokens = pred_lower.split()
    gt_tokens = gt_lower.split()
    common = set(pred_tokens) & set(gt_tokens)

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> bool:
    """
    Check if prediction exactly matches ground truth (case-insensitive, stripped).

    Args:
        prediction: The predicted answer string.
        ground_truth: The gold-standard answer string.

    Returns:
        True if exact match, False otherwise.
    """
    return prediction.strip().lower() == ground_truth.strip().lower()


def evaluate_on_squad(
    qa_pairs: list[dict],
    collection,
    session: SessionMemory,
) -> dict:
    """
    Run the Q&A system against SQuAD-formatted examples and compute metrics.

    Args:
        qa_pairs: List of dicts with 'question' and 'answer' keys.
        collection: ChromaDB collection with ingested documents.
        session: SessionMemory instance.

    Returns:
        Dict with: mean_f1, exact_match_rate, n_evaluated.
    """
    f1_scores_list: list[float] = []
    em_scores_list: list[int] = []

    for i, pair in enumerate(qa_pairs):
        try:
            result = answer_question(pair["question"], collection, session)
            f1_val = f1_score(result["answer"], pair["answer"])
            em_val = int(exact_match(result["answer"], pair["answer"]))
            f1_scores_list.append(f1_val)
            em_scores_list.append(em_val)

            if (i + 1) % 10 == 0:
                print(f"[eval] Processed {i+1}/{len(qa_pairs)}")
        except Exception as e:
            print(f"[eval] Error on example {i}: {e}")
            f1_scores_list.append(0.0)
            em_scores_list.append(0)

    return {
        "mean_f1": sum(f1_scores_list) / len(f1_scores_list) if f1_scores_list else 0.0,
        "exact_match_rate": sum(em_scores_list) / len(em_scores_list) if em_scores_list else 0.0,
        "n_evaluated": len(qa_pairs),
    }


# ─────────────────────────────────────────────
# VECTOR DB SETUP
# ─────────────────────────────────────────────

def init_vector_db(path: str = CHROMA_PATH):
    """
    Initialise ChromaDB with three collections:
      - document_chunks: the main retrieval store
      - interaction_history: past queries + which chunks were used
      - qa_cache: pre-computed answers for frequent queries

    Args:
        path: Directory path for ChromaDB persistent storage.

    Returns:
        Tuple of (client, chunks_col, interactions_col, cache_col).
    """
    client = chromadb.PersistentClient(path=path)

    chunks_col = client.get_or_create_collection("document_chunks")
    interactions_col = client.get_or_create_collection("interaction_history")
    cache_col = client.get_or_create_collection("qa_cache")

    return client, chunks_col, interactions_col, cache_col


# ─────────────────────────────────────────────
# MAIN — INTERACTIVE CLI DEMO
# ─────────────────────────────────────────────

def main():
    """Interactive CLI loop for the Document Q&A system."""
    print("=== Document Q&A System (RAG + Memory) ===\n")

    _, chunks_col, interactions_col, cache_col = init_vector_db()
    session = SessionMemory()
    ltm = LongTermMemory()

    # Ingest documents
    sample_docs: list[str] = []  # add file paths here, e.g. ["./data/paper.pdf"]
    for path in sample_docs:
        ingest_document(path, chunks_col)

    print("Type your question (or 'quit' to exit):\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        result = answer_question(
            query, chunks_col, session, ltm,
            interactions_col=interactions_col,
            cache_col=cache_col,
        )
        print(f"\nAssistant: {result['answer']}\n")

        # Show source attribution
        print("Sources:")
        for i, src in enumerate(result["sources"][:3], 1):
            print(f"  [{i}] {src['metadata'].get('source', 'unknown')} "
                  f"(score={src['score']:.2f})")
        print()

        # Collect feedback
        collect_feedback(query, result["answer"], ltm)
        print()


if __name__ == "__main__":
    main()
