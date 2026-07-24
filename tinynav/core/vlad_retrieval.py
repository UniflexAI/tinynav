import numpy as np


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
