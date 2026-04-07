"""
Centralized configuration for the RAG Document Q&A system.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")

# ── Paths ──
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
CHROMA_PATH: str = str(PROJECT_ROOT / "chroma_store_v2")
LTM_STORE_PATH: str = str(PROJECT_ROOT / "long_term_memory.json")
DATA_RAW_PATH: str = str(PROJECT_ROOT / "data" / "raw")
DATA_PROCESSED_PATH: str = str(PROJECT_ROOT / "data" / "processed")

# ── Chunking ──
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# ── Retrieval ──
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
RELEVANCE_WEIGHT: float = float(os.getenv("RELEVANCE_WEIGHT", "0.7"))
RECENCY_WEIGHT: float = float(os.getenv("RECENCY_WEIGHT", "0.3"))

# ── Memory ──
SHORT_TERM_LIMIT: int = int(os.getenv("SHORT_TERM_LIMIT", "20"))
CACHE_TTL_HOURS: float = float(os.getenv("CACHE_TTL_HOURS", "1.0"))
CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.97"))

# ── Context ──
MAX_CONTEXT_WORDS: int = int(os.getenv("MAX_CONTEXT_WORDS", "300"))
DOC_SUMMARY_WORDS: int = int(os.getenv("DOC_SUMMARY_WORDS", "2000"))

# ── Models ──
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "google/flan-t5-base")

# ── Server ──
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
