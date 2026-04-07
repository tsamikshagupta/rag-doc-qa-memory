"""
Local embeddings using sentence-transformers.
"""
import torch
import logging
from sentence_transformers import SentenceTransformer
from backend.utils.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading SentenceTransformer '{EMBEDDING_MODEL}' on {device}...")
model = SentenceTransformer(EMBEDDING_MODEL, device=device)

def get_embedding(text: str, task_type: str = "retrieval_document") -> list[float]:
    """
    Generate an embedding vector using a local SentenceTransformer model.
    """
    with torch.no_grad():
        embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()
