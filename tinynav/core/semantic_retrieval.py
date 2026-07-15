"""Small helpers for semantic/retrieval embeddings."""

from __future__ import annotations

import numpy as np


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Return an L2-normalized float32 embedding, preserving zero vectors."""
    arr = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm
