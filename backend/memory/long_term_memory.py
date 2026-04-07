"""
Long-term memory system with topic classification and semantic retrieval.
Persists Q&A pairs, corrections, and interaction patterns.
"""

import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb

from backend.utils.config import CHROMA_PATH, LTM_STORE_PATH
from backend.embeddings.sentence_transformer import get_embedding
from backend.llm.flan_t5 import generate_text

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    Persistent memory that stores successful Q&A pairs with topic labels,
    user corrections, and interaction history. Uses a ChromaDB collection
    (ltm_embeddings) for semantic retrieval of past Q&A pairs.

    Args:
        store_path: Path to the JSON persistence file.
        chroma_path: Path for ChromaDB persistent storage.
    """

    VALID_TOPICS = ("legal", "medical", "finance", "technical", "general")

    def __init__(
        self,
        store_path: str = LTM_STORE_PATH,
        chroma_path: str = CHROMA_PATH,
    ):
        self.store_path = store_path
        self.data: dict = {
            "qa_pairs": [],
            "corrections": [],
            "interaction_history": [],
        }
        self._load()
        client = chromadb.PersistentClient(path=chroma_path)
        self.ltm_collection = client.get_or_create_collection("ltm_embeddings")

    def _load(self) -> None:
        """Load persisted data from JSON."""
        if Path(self.store_path).exists():
            with open(self.store_path) as f:
                self.data = json.load(f)

    def _save(self) -> None:
        """Persist data to JSON."""
        with open(self.store_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def _classify_topic(self, query: str) -> str:
        """Use Gemini to classify query topic."""
        try:
            prompt = (
                "Classify the following query into exactly ONE topic: "
                "legal, medical, finance, technical, general.\n\n"
                f"Query: {query}\n\nTopic:"
            )
            topic = generate_text(prompt, max_length=15).strip().lower()
            return topic if topic in self.VALID_TOPICS else "general"
        except Exception as e:
            logger.warning(f"Topic classification failed: {e}")
            return "general"

    def record_feedback(
        self,
        query: str,
        answer: str,
        rating: int,
        correction: Optional[str] = None,
    ) -> None:
        """
        Record user feedback on a Q&A interaction.

        Classifies the topic via Gemini and stores the embedding in ChromaDB.
        If rating <= 2 and correction is given, stores the correction.

        Args:
            query: The original user query.
            answer: The system's answer.
            rating: User rating 1-5.
            correction: Optional better answer.
        """
        topic = self._classify_topic(query)
        now = datetime.now(timezone.utc).isoformat()

        entry = {
            "query": query,
            "answer": answer,
            "rating": rating,
            "topic": topic,
            "timestamp": now,
        }
        self.data["qa_pairs"].append(entry)

        if correction and rating <= 2:
            self.data["corrections"].append({
                "query": query,
                "original_answer": answer,
                "correction": correction,
                "timestamp": now,
            })

        self._save()

        # Store embedding for semantic retrieval
        try:
            qa_text = f"Q: {query}\nA: {answer}"
            embedding = get_embedding(qa_text)
            ltm_id = f"ltm_{hashlib.md5(qa_text.encode()).hexdigest()[:12]}"
            self.ltm_collection.upsert(
                ids=[ltm_id],
                embeddings=[embedding],
                documents=[json.dumps(entry)],
                metadatas=[ {"query": query[:200], "rating": rating, "topic": topic, "timestamp": now}],
            )
        except Exception as e:
            logger.error(f"Failed to store LTM embedding: {e}")

    def retrieve_relevant(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve semantically similar past Q&A pairs with rating >= 3.

        Args:
            query: Current user query.
            top_k: Max results.

        Returns:
            List of Q&A pair dicts.
        """
        if self.ltm_collection.count() > 0:
            try:
                qe = get_embedding(query, task_type="retrieval_query")
                results = self.ltm_collection.query(
                    query_embeddings=[qe],
                    n_results=top_k * 2,
                    include=["documents", "metadatas", "distances"],
                    where={"rating": {"$gte": 3}},
                )
                pairs: list[dict] = []
                for doc_str in results["documents"][0]:
                    try:
                        pairs.append(json.loads(doc_str))
                    except json.JSONDecodeError:
                        continue
                return pairs[:top_k]
            except Exception as e:
                logger.warning(f"Semantic LTM retrieval failed: {e}")

        # Fallback
        high_rated = [p for p in self.data["qa_pairs"] if p.get("rating", 0) >= 3]
        return high_rated[-top_k:]

    def get_stats(self) -> dict:
        """Return memory statistics."""
        return {
            "total_qa_pairs": len(self.data["qa_pairs"]),
            "total_corrections": len(self.data["corrections"]),
            "ltm_embeddings": self.ltm_collection.count(),
            "topics": {t: sum(1 for p in self.data["qa_pairs"] if p.get("topic") == t) for t in self.VALID_TOPICS},
        }
