"""DINOv2 Patch VLAD (Vector of Locally Aggregated Descriptors).

Replaces the CLS-token cosine-similarity retrieval in tinynav with a patch-level
place-recognition descriptor inspired by AnyLoc.

Pipeline:
  1. Extract DINOv2 patch tokens (N_patch, C) per image — already computed by TRT.
  2. Train a K-means vocabulary on all patch tokens from the map.
  3. For each image, compute a VLAD descriptor by assigning patches to clusters
     and aggregating residuals.
  4. Cosine similarity between VLAD descriptors → top-k retrieval.

All routines are pure numpy / scipy so they run on both x64 and Jetson aarch64.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def train_vocabulary(
    patch_tokens: np.ndarray,
    vocab_size: int = 32,
    iterations: int = 200,
    batch_size: int = 1024,
    seed: int = 42,
) -> np.ndarray:
    """Train a MiniBatch K-means vocabulary on DINOv2 patch tokens.

    Args:
        patch_tokens: (M, C) stacked patch tokens from all keyframes.
        vocab_size: number of cluster centres (K).
        iterations: minibatch k-means iterations.
        batch_size: minibatch size.
        seed: RNG seed for reproducibility.

    Returns:
        centres: (vocab_size, C) L2-normalised cluster centres.
    """
    rng = np.random.default_rng(seed)
    M = patch_tokens.shape[0]
    if M < vocab_size:
        raise ValueError(f"Need at least {vocab_size} tokens, got {M}")

    # L2-normalise tokens before clustering (DINOv2 patch tokens are already
    # normalised in practice, but enforce it here).
    tokens = patch_tokens.astype(np.float32, copy=True)
    norms = np.linalg.norm(tokens, axis=1, keepdims=True)
    tokens /= np.maximum(norms, 1e-8)

    centres = tokens[rng.choice(M, size=vocab_size, replace=False)].copy()
    counts = np.zeros(vocab_size, dtype=np.int64)

    for it in range(iterations):
        batch = tokens[rng.choice(M, size=min(batch_size, M), replace=False)]
        labels = cKDTree(centres).query(batch, k=1, workers=-1)[1]
        for label, vec in zip(labels, batch):
            counts[label] += 1
            eta = 1.0 / counts[label]
            centres[label] = (1.0 - eta) * centres[label] + eta * vec
        if (it + 1) % 50 == 0:
            logger.info(f"VLAD k-means iteration {it + 1}/{iterations}")

    norms = np.linalg.norm(centres, axis=1, keepdims=True)
    centres = (centres / np.maximum(norms, 1e-8)).astype(np.float32)
    return centres


def compute_vlad(patch_tokens: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Compute a single-image VLAD descriptor.

    Args:
        patch_tokens: (N, C) patch tokens for one image (L2-normalised).
        centres: (K, C) vocabulary centres (L2-normalised).

    Returns:
        descriptor: (K * C,) L2-normalised VLAD descriptor.
    """
    K, C = centres.shape
    if len(patch_tokens) == 0:
        return np.zeros(K * C, dtype=np.float32)

    tokens = patch_tokens.astype(np.float32, copy=False)
    norms = np.linalg.norm(tokens, axis=1, keepdims=True)
    tokens /= np.maximum(norms, 1e-8)

    tree = cKDTree(centres)
    labels = tree.query(tokens, k=1, workers=-1)[1]

    residuals = np.zeros_like(centres, dtype=np.float32)
    for k in range(K):
        mask = labels == k
        if np.any(mask):
            residuals[k] = np.sum(tokens[mask] - centres[k], axis=0)

    # Intra-normalisation.
    row_norms = np.linalg.norm(residuals, axis=1, keepdims=True)
    residuals /= np.maximum(row_norms, 1e-8)

    descriptor = residuals.reshape(-1)
    desc_norm = np.linalg.norm(descriptor)
    if desc_norm > 1e-8:
        descriptor /= desc_norm
    return descriptor.astype(np.float32)


def compute_vlad_batch(
    patch_tokens_list: list[np.ndarray],
    centres: np.ndarray,
) -> np.ndarray:
    """Compute VLAD descriptors for a list of images.

    Args:
        patch_tokens_list: list of (N_i, C) patch token arrays.
        centres: (K, C) vocabulary centres.

    Returns:
        descriptors: (len(list), K * C) L2-normalised VLAD descriptors.
    """
    K, C = centres.shape
    descriptors = np.zeros((len(patch_tokens_list), K * C), dtype=np.float32)
    for i, tokens in enumerate(patch_tokens_list):
        descriptors[i] = compute_vlad(tokens, centres)
        if (i + 1) % 100 == 0:
            logger.info(f"VLAD encoded {i + 1}/{len(patch_tokens_list)}")
    return descriptors


def find_loop_vlad(
    query_vlad: np.ndarray,
    map_vlads: np.ndarray,
    similarity_threshold: float,
    top_k: int,
) -> list[tuple[int, float]]:
    """Find top-k loop closure candidates using VLAD cosine similarity.

    Args:
        query_vlad: (D,) L2-normalised VLAD descriptor.
        map_vlads: (N, D) L2-normalised VLAD descriptors.
        similarity_threshold: minimum cosine similarity.
        top_k: maximum number of candidates.

    Returns:
        List of (index, similarity) tuples, sorted ascending (best last).
    """
    if len(map_vlads) == 0:
        return []
    similarity_array = map_vlads @ query_vlad  # both are L2-normalised → dot = cosine
    top_k_indices = np.argsort(similarity_array, axis=0)
    loop_list = []
    for idx in top_k_indices:
        if similarity_array[idx] > similarity_threshold:
            loop_list.append((int(idx), float(similarity_array[idx])))
    return loop_list[-top_k:]
