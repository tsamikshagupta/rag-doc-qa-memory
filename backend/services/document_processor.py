"""
Phase 1 — Document Processing Pipeline.
Handles loading, chunking, embedding, and ingestion of documents.
"""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.utils.config import CHUNK_SIZE, CHUNK_OVERLAP, DOC_SUMMARY_WORDS
from backend.embeddings.sentence_transformer import get_embedding

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DOCUMENT LOADING
# ─────────────────────────────────────────────

def load_document(file_path: str) -> str:
    """
    Load and extract raw text from a document file.

    Supported: PDF (.pdf), DOCX (.docx), HTML (.html/.htm), TXT (.txt), MD (.md).

    Args:
        file_path: Path to the document file.

    Returns:
        The full extracted text as a single string.

    Raises:
        FileNotFoundError: If file does not exist.
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
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)

    elif ext == ".docx":
        from docx import Document
        doc = Document(file_path)
        paragraphs: list[str] = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    elif ext in (".html", ".htm"):
        from bs4 import BeautifulSoup
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into overlapping chunks using paragraph-first + sliding-window fallback.

    Strategy:
      1. Split on double-newlines (paragraph boundaries).
      2. Merge short adjacent paragraphs up to chunk_size words.
      3. Subdivide oversized blocks with a sliding window.

    Returns:
        List of dicts with keys: text, chunk_index, char_start, char_end.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    raw_blocks: list[str] = []
    current_block: list[str] = []
    current_wc = 0

    for para in paragraphs:
        pw = len(para.split())
        if current_wc + pw <= chunk_size:
            current_block.append(para)
            current_wc += pw
        else:
            if current_block:
                raw_blocks.append("\n\n".join(current_block))
            current_block = [para]
            current_wc = pw
    if current_block:
        raw_blocks.append("\n\n".join(current_block))

    final_texts: list[str] = []
    for block in raw_blocks:
        words = block.split()
        if len(words) <= chunk_size:
            final_texts.append(block)
        else:
            step = max(1, chunk_size - overlap)
            for start in range(0, len(words), step):
                window = words[start:start + chunk_size]
                final_texts.append(" ".join(window))
                if start + chunk_size >= len(words):
                    break

    chunks: list[dict] = []
    search_start = 0
    for idx, ct in enumerate(final_texts):
        anchor = ct[:80]
        cs = text.find(anchor, search_start)
        if cs == -1:
            cs = search_start
        ce = cs + len(ct)
        chunks.append({"text": ct, "chunk_index": idx, "char_start": cs, "char_end": ce})
        search_start = max(search_start, cs + 1)

    return chunks


# ─────────────────────────────────────────────
# INGESTION
# ─────────────────────────────────────────────

def ingest_document(file_path: str, collection) -> str:
    """
    Full ingestion: load → chunk → embed → store in ChromaDB.

    Also creates a doc-level summary embedding (first 2000 words).

    Args:
        file_path: Path to the document.
        collection: ChromaDB collection to store into.

    Returns:
        The generated doc_id string.
    """
    doc_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
    raw_text = load_document(file_path)
    if not raw_text.strip():
        raise ValueError("No text could be extracted from this document. It might be an image-only PDF.")
    chunks = chunk_text(raw_text)
    ext = Path(file_path).suffix.lower()
    now = datetime.now(timezone.utc).isoformat()

    for chunk in chunks:
        embedding = get_embedding(chunk["text"])
        chunk_id = f"{doc_id}_chunk_{chunk['chunk_index']}"
        page_number = chunk["char_start"] // 3000 + 1 if ext == ".pdf" else -1

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

    # Document-level summary
    summary_text = " ".join(raw_text.split()[:DOC_SUMMARY_WORDS])
    summary_emb = get_embedding(summary_text)
    collection.add(
        ids=[f"{doc_id}_summary"],
        embeddings=[summary_emb],
        documents=[summary_text],
        metadatas=[{
            "doc_id": doc_id, "source": file_path, "chunk_index": -1,
            "char_start": 0, "char_end": len(summary_text),
            "page_number": -1, "ingested_at": now, "type": "doc_summary",
        }],
    )

    logger.info(f"Ingested {file_path} → {len(chunks)} chunks + 1 summary (doc_id={doc_id})")
    return doc_id
