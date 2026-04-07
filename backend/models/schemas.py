"""
Pydantic request/response models for the FastAPI endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for POST /ask."""
    query: str = Field(..., description="The user's question")
    session_id: str = Field(default="default", description="Session identifier for multi-turn")


class AskResponse(BaseModel):
    """Response body for POST /ask."""
    answer: str
    sources: list[dict]
    query: str


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback."""
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    correction: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response body for POST /feedback."""
    status: str
    rating: int
    correction_stored: bool


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str
    collections: dict


class MemoryStatsResponse(BaseModel):
    """Response body for GET /memory."""
    total_qa_pairs: int
    total_corrections: int
    ltm_embeddings: int
    topics: dict


class MetricsResponse(BaseModel):
    """Response body for GET /metrics."""
    total_documents: int
    total_interactions: int
    cached_queries: int
    memory_stats: dict
