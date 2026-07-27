import numpy as np


def train_vocabulary_streaming(
    batch_iterator_factory,
    num_clusters: int,
    epochs: int = 1,
    batch_size: int = 1024,
    seed: int = 0,
) -> np.ndarray:
    """Fit a VLAD codebook via streamed online k-means instead of Lloyd's k-means over a
    pooled, in-memory sample of patch descriptors.

    ``fit_vlad_codebook`` requires every training descriptor concatenated into one array
    up front; on a large map, pooling even a bounded keyframe sample (keyframes x patches
    x dims, plus k-means' own working memory) is multiple GB and risks OOM. This instead
    calls ``batch_iterator_factory()`` once per epoch to get a fresh iterator over
    per-keyframe (N_i, D) descriptor arrays (e.g. re-run inference or a freshly shuffled
    disk read), concatenates them into ``batch_size`` chunks on the fly, and folds each
    chunk into the centres via a decaying running-mean update (Robbins-Monro / sequential
    k-means, https://gist.github.com/yjzhang/aaf460849a4398422785c0e85932688d) instead of
    holding the full training pool in memory at once. Operates in the same raw (non
    L2-normalised) descriptor space as ``fit_vlad_codebook`` so the resulting codebook is
    a drop-in replacement for ``compute_vlad`` below.

    Args:
        batch_iterator_factory: zero-arg callable returning a fresh iterator of
            (N_i, D) patch-descriptor arrays; called once per epoch.
        num_clusters: number of cluster centres (K).
        epochs: number of streamed passes over the data.
        batch_size: size of each online-update chunk.
        seed: RNG seed for reproducibility.

    Returns:
        centres: (num_clusters, D) cluster centres.
    """
    rng = np.random.default_rng(seed)
    centres = None
    counts = None

    def apply_batch(batch: np.ndarray) -> None:
        nonlocal centres, counts
        if centres is None:
            if len(batch) < num_clusters:
                raise ValueError(f"Need at least {num_clusters} descriptors in the first batch, got {len(batch)}")
            centres = batch[rng.choice(len(batch), size=num_clusters, replace=False)].copy()
            counts = np.zeros(num_clusters, dtype=np.int64)
        distances = (
            np.sum(batch ** 2, axis=1, keepdims=True)
            + np.sum(centres ** 2, axis=1)
            - 2.0 * batch @ centres.T
        )
        labels = np.argmin(distances, axis=1)
        for label, vec in zip(labels, batch):
            counts[label] += 1
            eta = 1.0 / counts[label]
            centres[label] = (1.0 - eta) * centres[label] + eta * vec

    for _epoch in range(epochs):
        pending: list[np.ndarray] = []
        pending_count = 0
        for frame_descriptors in batch_iterator_factory():
            pending.append(np.asarray(frame_descriptors, dtype=np.float32))
            pending_count += len(frame_descriptors)
            while pending_count >= batch_size:
                buf = np.concatenate(pending, axis=0)
                batch, rest = buf[:batch_size], buf[batch_size:]
                pending = [rest] if len(rest) else []
                pending_count = len(rest)
                apply_batch(batch)
        if pending_count > 0:
            apply_batch(np.concatenate(pending, axis=0))

    return centres.astype(np.float32)


def fit_vlad_codebook(patch_descriptors: np.ndarray, num_clusters: int, num_iters: int = 25, seed: int = 0) -> np.ndarray:
    """Fit a VLAD codebook (cluster centers) over a pool of patch descriptors via Lloyd's k-means.

    patch_descriptors: (N, D) descriptors pooled across many keyframes.
    Returns: (num_clusters, D) cluster centers.
    """
    patch_descriptors = np.asarray(patch_descriptors, dtype=np.float32)
    assert patch_descriptors.ndim == 2
    assert num_clusters > 0
    assert patch_descriptors.shape[0] >= num_clusters

    rng = np.random.default_rng(seed)
    init_indices = rng.choice(patch_descriptors.shape[0], size=num_clusters, replace=False)
    centers = patch_descriptors[init_indices].copy()

    for _ in range(num_iters):
        # (N, K) squared distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        distances = (
            np.sum(patch_descriptors ** 2, axis=1, keepdims=True)
            + np.sum(centers ** 2, axis=1)
            - 2.0 * patch_descriptors @ centers.T
        )
        assignments = np.argmin(distances, axis=1)

        new_centers = centers.copy()
        for k in range(num_clusters):
            members = patch_descriptors[assignments == k]
            if len(members) > 0:
                new_centers[k] = members.mean(axis=0)
        centers = new_centers

    return centers


def compute_vlad(patch_descriptors: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Aggregate patch descriptors into a single VLAD vector given a fitted codebook.

    patch_descriptors: (N, D) descriptors for a single image.
    codebook: (K, D) cluster centers.
    Returns: (K * D,) L2-normalized VLAD vector.
    """
    patch_descriptors = np.asarray(patch_descriptors, dtype=np.float32)
    codebook = np.asarray(codebook, dtype=np.float32)
    assert patch_descriptors.ndim == 2
    assert codebook.ndim == 2
    assert patch_descriptors.shape[1] == codebook.shape[1]

    num_clusters, dim = codebook.shape
    distances = (
        np.sum(patch_descriptors ** 2, axis=1, keepdims=True)
        + np.sum(codebook ** 2, axis=1)
        - 2.0 * patch_descriptors @ codebook.T
    )
    assignments = np.argmin(distances, axis=1)

    vlad = np.zeros((num_clusters, dim), dtype=np.float32)
    for k in range(num_clusters):
        members = patch_descriptors[assignments == k]
        if len(members) > 0:
            residuals = members - codebook[k]
            vlad[k] = residuals.sum(axis=0)

    # Intra-normalization: signed sqrt then L2-normalize each cluster's residual.
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    cluster_norms = np.linalg.norm(vlad, axis=1, keepdims=True)
    cluster_norms[cluster_norms == 0.0] = 1.0
    vlad = vlad / cluster_norms

    vlad = vlad.reshape(-1)
    norm = np.linalg.norm(vlad)
    if norm > 0.0:
        vlad = vlad / norm
    return vlad
