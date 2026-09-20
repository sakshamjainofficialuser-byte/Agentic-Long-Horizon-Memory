"""Unit tests for FAISS vector store and semantic memory retrieval."""

import os
import tempfile
import numpy as np
import pytest
import memory.retrieval as retrieval_module
from memory.retrieval import EMBEDDING_DIM, FAISSMemoryStore


def mock_deterministic_embedding(text: str) -> np.ndarray:
    """Generate consistent deterministic vector based on text hash for test isolation."""
    np.random.seed(abs(hash(text)) % (2**32))
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm if norm > 0 else 1.0)


@pytest.fixture(autouse=True)
def mock_embedding_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure all retrieval unit tests run offline with mock embeddings."""
    monkeypatch.setattr(retrieval_module, "get_embedding", mock_deterministic_embedding)


@pytest.fixture
def temp_faiss_store() -> FAISSMemoryStore:
    """Create a temporary FAISS store isolated from persistent files."""
    fd_idx, path_idx = tempfile.mkstemp(suffix=".bin")
    fd_meta, path_meta = tempfile.mkstemp(suffix=".json")
    os.close(fd_idx)
    os.close(fd_meta)

    store = FAISSMemoryStore(index_path=path_idx, meta_path=path_meta)
    yield store

    if os.path.exists(path_idx):
        os.remove(path_idx)
    if os.path.exists(path_meta):
        os.remove(path_meta)


def test_faiss_add_and_search(temp_faiss_store: FAISSMemoryStore) -> None:
    """Verify adding memories to FAISS and retrieving relevant content."""
    temp_faiss_store.add_memory("User favorite language is Rust", session_id="sess_1")
    temp_faiss_store.add_memory("Project deadline is next Friday", session_id="sess_1")

    assert temp_faiss_store.index.ntotal == 2

    # Query with low min_score to test retrieval mechanics
    results = temp_faiss_store.search("Project deadline is next Friday", session_id="sess_1", limit=1, min_score=0.0)
    assert len(results) == 1
    assert "deadline" in results[0].lower()


def test_faiss_session_filtering(temp_faiss_store: FAISSMemoryStore) -> None:
    """Verify FAISS search honors session_id filtering."""
    temp_faiss_store.add_memory("Memory for Alpha session", session_id="sess_alpha")
    temp_faiss_store.add_memory("Memory for Beta session", session_id="sess_beta")

    results_alpha = temp_faiss_store.search("Memory for Alpha session", session_id="sess_alpha", min_score=0.0)
    assert len(results_alpha) >= 1
    assert all("Alpha" in m for m in results_alpha)

    results_beta = temp_faiss_store.search("Memory for Beta session", session_id="sess_beta", min_score=0.0)
    assert len(results_beta) >= 1
    assert all("Beta" in m for m in results_beta)


def test_faiss_empty_search(temp_faiss_store: FAISSMemoryStore) -> None:
    """Verify search returns empty list gracefully on empty index or blank query."""
    assert temp_faiss_store.search("Anything") == []
    assert temp_faiss_store.search("") == []
