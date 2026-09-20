"""FAISS Vector Store for semantic memory indexing and similarity retrieval."""

import json
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import faiss
import numpy as np
import openai
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536
DEFAULT_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
DEFAULT_META_PATH = os.getenv("FAISS_META_PATH", "faiss_meta.json")


def get_embedding(text: str) -> np.ndarray:
    """Generate normalized embedding vector using OpenAI Embeddings API.

    Returns
    -------
    np.ndarray
        1D float32 numpy array with shape (1536,), L2 normalized for cosine similarity.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Fallback dummy embedding if key is absent (e.g. in offline unit tests)
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 0 else 1.0)

    try:
        client = OpenAI(api_key=api_key, timeout=3.0, max_retries=1)
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 0 else 1.0)
    except Exception:
        # Graceful fallback on network/quota error
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 0 else 1.0)


class FAISSMemoryStore:
    """Vector index manager backed by FAISS IndexFlatIP (cosine similarity)."""

    def __init__(
        self,
        dimension: int = EMBEDDING_DIM,
        index_path: str = DEFAULT_INDEX_PATH,
        meta_path: str = DEFAULT_META_PATH,
    ) -> None:
        self.dimension = dimension
        self.index_path = index_path
        self.meta_path = meta_path
        self.metadata: list[dict[str, Any]] = []

        # Use IndexFlatIP (Inner Product) since embeddings are L2 normalized
        self.index = faiss.IndexFlatIP(self.dimension)
        self._load_from_disk()

    def add_memory(
        self,
        text: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Embed text and insert it into the FAISS index with metadata."""
        cleaned_text = text.strip()
        if not cleaned_text:
            return

        vector = get_embedding(cleaned_text).reshape(1, -1)
        self.index.add(vector)

        meta_entry = {
            "text": cleaned_text,
            "session_id": session_id,
            "extra": metadata or {},
        }
        self.metadata.append(meta_entry)
        self._save_to_disk()

    def search(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 5,
        min_score: float = 0.2,
    ) -> list[str]:
        """Search top-K relevant memories by cosine similarity.

        Parameters
        ----------
        query : str
            Query text to embed and match against stored memories.
        session_id : str | None, optional
            Filter memories to a specific session (or match cross-session if None).
        limit : int, default=5
            Maximum number of memory snippets to return.
        min_score : float, default=0.2
            Minimum cosine similarity threshold.

        Returns
        -------
        list[str]
            List of matched memory text snippets.
        """
        if self.index.ntotal == 0 or not query.strip():
            return []

        query_vec = get_embedding(query.strip()).reshape(1, -1)
        k = min(self.index.ntotal, limit * 3)  # fetch buffer to allow session filtering
        scores, indices = self.index.search(query_vec, k)

        results: list[str] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            if score < min_score:
                continue

            entry = self.metadata[idx]
            # Match session if specified
            if session_id is None or entry["session_id"] == session_id:
                results.append(entry["text"])
                if len(results) >= limit:
                    break

        return results

    def _save_to_disk(self) -> None:
        """Persist FAISS index and metadata to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_from_disk(self) -> None:
        """Load FAISS index and metadata if saved files exist."""
        if Path(self.index_path).exists() and Path(self.meta_path).exists():
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = []


# Singleton instance for memory operations
_STORE_INSTANCE: FAISSMemoryStore | None = None


def get_memory_store() -> FAISSMemoryStore:
    """Return the shared singleton FAISS memory store instance."""
    global _STORE_INSTANCE
    if _STORE_INSTANCE is None:
        _STORE_INSTANCE = FAISSMemoryStore()
    return _STORE_INSTANCE
