import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    timestamp: int
    score: float
    pose: np.ndarray | None = None
    rgb_image_loader: Callable[[], np.ndarray | None] | None = None
    infra1_image_loader: Callable[[], np.ndarray | None] | None = None


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("embedding norm must be finite and positive")
    return embedding / norm


def rank_semantic_embeddings(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    timestamps: list[int],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    if top_k <= 0 or len(timestamps) == 0:
        return []
    query_embedding = normalize_embedding(query_embedding)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be rank-2, got shape {embeddings.shape}")
    if embeddings.shape[0] != len(timestamps):
        raise ValueError(f"embedding count {embeddings.shape[0]} != timestamp count {len(timestamps)}")
    if embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError(f"embedding dim {embeddings.shape[1]} != query dim {query_embedding.shape[0]}")

    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(timestamps[int(idx)], float(scores[int(idx)])) for idx in top_indices]


def load_semantic_embedding_matrix(db, timestamps: list[int]) -> tuple[np.ndarray, list[int]]:
    valid_timestamps = []
    embeddings = []
    for timestamp in timestamps:
        if not db.has_semantic_embedding(timestamp):
            logger.warning("Missing semantic embedding for timestamp %s; skipping retrieval entry", timestamp)
            continue
        embeddings.append(normalize_embedding(db.get_semantic_embedding(timestamp)))
        valid_timestamps.append(timestamp)

    if len(embeddings) == 0:
        return np.empty((0, 0), dtype=np.float32), []
    return np.stack(embeddings).astype(np.float32), valid_timestamps
