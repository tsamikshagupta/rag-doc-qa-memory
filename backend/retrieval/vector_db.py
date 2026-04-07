"""
ChromaDB vector database initialization and management.
"""

import chromadb
from backend.utils.config import CHROMA_PATH


def init_vector_db(path: str = CHROMA_PATH):
    """
    Initialize ChromaDB with three collections:
      - document_chunks: main retrieval store
      - interaction_history: past queries + chunks used
      - qa_cache: pre-computed answers for repeated queries

    Args:
        path: Directory for ChromaDB persistent storage.

    Returns:
        Tuple of (client, chunks_col, interactions_col, cache_col).
    """
    client = chromadb.PersistentClient(path=path)
    chunks_col = client.get_or_create_collection("document_chunks")
    interactions_col = client.get_or_create_collection("interaction_history")
    cache_col = client.get_or_create_collection("qa_cache")
    return client, chunks_col, interactions_col, cache_col
