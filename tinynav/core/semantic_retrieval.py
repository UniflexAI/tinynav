import numpy as np


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
    assert top_k > 0
    query_embedding = normalize_embedding(query_embedding)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    assert embeddings.ndim == 2
    assert embeddings.shape == (len(timestamps), query_embedding.shape[0])

    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(timestamps[int(idx)], float(scores[int(idx)])) for idx in top_indices]


def load_semantic_embedding_matrix(db, timestamps: list[int]) -> tuple[np.ndarray, list[int]]:
    embeddings = []
    for timestamp in timestamps:
        assert db.has_semantic_embedding(timestamp)
        embeddings.append(normalize_embedding(db.get_semantic_embedding(timestamp)))

    if len(embeddings) == 0:
        return np.empty((0, 0), dtype=np.float32), []
    return np.stack(embeddings).astype(np.float32), timestamps
