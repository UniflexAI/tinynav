import numpy as np

from tinynav.core.semantic_retrieval import normalize_embedding, rank_semantic_embeddings


def test_normalize_embedding():
    embedding = normalize_embedding(np.array([3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(embedding, np.array([0.6, 0.8], dtype=np.float32))


def test_normalize_embedding_rejects_zero_vector():
    try:
        normalize_embedding(np.zeros(4, dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_rank_semantic_embeddings_returns_best_first():
    timestamps = [10, 20, 30]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
        ],
        dtype=np.float32,
    )
    ranked = rank_semantic_embeddings(np.array([1.0, 0.0], dtype=np.float32), embeddings, timestamps, top_k=2)
    assert [timestamp for timestamp, _score in ranked] == [10, 30]


def test_rank_semantic_embeddings_checks_dimensions():
    try:
        rank_semantic_embeddings(
            np.array([1.0, 0.0], dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            [1, 2],
        )
    except AssertionError:
        return
    raise AssertionError("expected AssertionError")


if __name__ == "__main__":
    test_normalize_embedding()
    test_normalize_embedding_rejects_zero_vector()
    test_rank_semantic_embeddings_returns_best_first()
    test_rank_semantic_embeddings_checks_dimensions()
