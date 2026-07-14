from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class BowConfig:
    vocab_size: int = 512
    max_desc_per_image_for_train: int = 40
    random_seed: int = 7
    kmeans_max_iter: int = 40
    kmeans_eps: float = 0.02


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def valid_superpoint_descriptors(features: dict) -> np.ndarray:
    """Return valid L2-normalized SuperPoint descriptors as ``[N, D]`` float32."""
    descriptors = features["descps"][0].astype(np.float32)
    mask = features.get("mask")
    if mask is not None:
        descriptors = descriptors[mask[0].reshape(-1).astype(bool)]
    if len(descriptors) == 0:
        return descriptors.reshape(0, features["descps"].shape[-1])
    return _normalize_rows(descriptors)


def _fit_vocab(descriptor_samples: np.ndarray, config: BowConfig) -> np.ndarray:
    vocab_size = min(config.vocab_size, len(descriptor_samples))
    if vocab_size < 2:
        raise ValueError(
            f"not enough descriptors to train BoW vocabulary: "
            f"{len(descriptor_samples)} < 2"
        )
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        config.kmeans_max_iter,
        config.kmeans_eps,
    )
    _compactness, _labels, centers = cv2.kmeans(
        descriptor_samples.astype(np.float32),
        vocab_size,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    return _normalize_rows(centers.astype(np.float32))


def assign_words(descriptors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if len(descriptors) == 0:
        return np.empty((0,), dtype=np.int64)
    similarities = descriptors @ centers.T
    return np.argmax(similarities, axis=1).astype(np.int64)


def make_bow_vector(descriptors: np.ndarray, centers: np.ndarray, idf: np.ndarray) -> np.ndarray:
    words = assign_words(descriptors, centers)
    hist = np.bincount(words, minlength=len(centers)).astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    hist *= idf
    hist /= max(np.linalg.norm(hist), 1e-8)
    return hist.astype(np.float32)


def build_bow_index(
    db,
    timestamps: list[int],
    output_path: str | os.PathLike[str],
    config: BowConfig = BowConfig(),
) -> None:
    """Build a TF-IDF BoW index from map SuperPoint features."""
    rng = np.random.default_rng(config.random_seed)
    map_descriptors: list[np.ndarray] = []
    train_samples: list[np.ndarray] = []

    for timestamp in timestamps:
        features = db.features[timestamp]
        descriptors = valid_superpoint_descriptors(features)
        map_descriptors.append(descriptors)
        if len(descriptors) > config.max_desc_per_image_for_train:
            indices = rng.choice(len(descriptors), size=config.max_desc_per_image_for_train, replace=False)
            train_samples.append(descriptors[indices])
        else:
            train_samples.append(descriptors)

    descriptor_samples = np.concatenate(train_samples, axis=0).astype(np.float32)
    centers = _fit_vocab(descriptor_samples, config)
    vocab_size = len(centers)

    raw_hists = []
    df = np.zeros(vocab_size, dtype=np.float32)
    for descriptors in map_descriptors:
        words = assign_words(descriptors, centers)
        hist = np.bincount(words, minlength=vocab_size).astype(np.float32)
        raw_hists.append(hist)
        df += hist > 0

    idf = (np.log((1 + len(timestamps)) / (1 + df)) + 1.0).astype(np.float32)
    bow_vectors = []
    for hist in raw_hists:
        if hist.sum() > 0:
            hist = hist / hist.sum()
        vector = hist * idf
        vector = vector / max(np.linalg.norm(vector), 1e-8)
        bow_vectors.append(vector.astype(np.float32))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        timestamps=np.asarray(timestamps, dtype=np.int64),
        centers=centers,
        idf=idf,
        bow_vectors=np.stack(bow_vectors).astype(np.float32),
        vocab_size=np.asarray([vocab_size], dtype=np.int64),
    )


class BowIndex:
    def __init__(self, path: str | os.PathLike[str]):
        data = np.load(path)
        self.timestamps = data["timestamps"].astype(np.int64)
        self.centers = data["centers"].astype(np.float32)
        self.idf = data["idf"].astype(np.float32)
        self.bow_vectors = data["bow_vectors"].astype(np.float32)

    def query(self, features: dict, top_k: int) -> list[tuple[int, float]]:
        descriptors = valid_superpoint_descriptors(features)
        query_vector = make_bow_vector(descriptors, self.centers, self.idf)
        similarities = self.bow_vectors @ query_vector
        if len(similarities) == 0:
            return []
        top_k = min(top_k, len(similarities))
        candidate_indices = np.argpartition(-similarities, top_k - 1)[:top_k]
        candidate_indices = candidate_indices[np.argsort(-similarities[candidate_indices])]
        return [(int(self.timestamps[idx]), float(similarities[idx])) for idx in candidate_indices]
