import cv2
import numpy as np


SUPERPOINT_BOW_INDEX_FILENAME = "superpoint_bow_index.npz"


class SuperPointBoWRetriever:
    """Lightweight SuperPoint Bag-of-Words retrieval index for map keyframes."""

    def __init__(
        self,
        vocab_size: int = 512,
        sample_limit: int = 120000,
        kmeans_iterations: int = 30,
    ):
        self.vocab_size = vocab_size
        self.sample_limit = sample_limit
        self.kmeans_iterations = kmeans_iterations
        self.vocab: np.ndarray | None = None
        self.histograms: np.ndarray | None = None
        self.idf: np.ndarray | None = None
        self.timestamps: list[int] = []

    @staticmethod
    def descriptors_from_features(features: dict) -> np.ndarray:
        descriptors = np.asarray(features["descps"])
        mask = np.asarray(features.get("mask", []))

        descriptors = np.squeeze(descriptors)
        if descriptors.ndim != 2:
            return np.empty((0, 0), dtype=np.float32)

        valid_count = None
        if mask.size > 0:
            valid = np.squeeze(mask).astype(bool)
            valid_count = int(valid.shape[0])
        else:
            valid = None

        if valid_count is not None:
            if descriptors.shape[0] == valid_count:
                desc = descriptors
            elif descriptors.shape[1] == valid_count:
                desc = descriptors.T
            else:
                desc = descriptors if descriptors.shape[0] >= descriptors.shape[1] else descriptors.T
        else:
            desc = descriptors if descriptors.shape[0] >= descriptors.shape[1] else descriptors.T

        if valid is not None and len(valid) == len(desc):
            desc = desc[valid]

        desc = np.asarray(desc, dtype=np.float32)
        if desc.size == 0:
            return desc.reshape(0, 0)
        norms = np.linalg.norm(desc, axis=1, keepdims=True)
        return desc / np.maximum(norms, 1e-6)

    def build_from_feature_loader(self, timestamps: list[int], feature_loader) -> None:
        self.timestamps = []
        sampled_descriptors = []
        rng = np.random.default_rng(0)
        sampled_count = 0

        for timestamp in timestamps:
            features = feature_loader(timestamp)
            descriptors = self.descriptors_from_features(features)
            if descriptors.size == 0:
                continue
            self.timestamps.append(timestamp)

            remaining = self.sample_limit - sampled_count
            if remaining <= 0:
                continue
            take = min(len(descriptors), max(1, remaining // max(1, len(timestamps))))
            if take < len(descriptors):
                sample_indices = rng.choice(len(descriptors), size=take, replace=False)
                sampled = descriptors[sample_indices]
            else:
                sampled = descriptors
            sampled_descriptors.append(sampled)
            sampled_count += len(sampled)

        if len(sampled_descriptors) == 0:
            raise RuntimeError("No SuperPoint descriptors found for BoW relocalization")

        samples = np.concatenate(sampled_descriptors, axis=0).astype(np.float32)
        if len(samples) > self.sample_limit:
            sample_indices = rng.choice(len(samples), size=self.sample_limit, replace=False)
            samples = samples[sample_indices]

        vocab_size = min(self.vocab_size, len(samples))
        if vocab_size < 2:
            raise RuntimeError(f"Not enough SuperPoint descriptors for BoW relocalization: {len(samples)}")

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            self.kmeans_iterations,
            1e-3,
        )
        _, _, centers = cv2.kmeans(
            samples,
            vocab_size,
            None,
            criteria,
            1,
            cv2.KMEANS_PP_CENTERS,
        )
        self.vocab = np.asarray(centers, dtype=np.float32)

        histograms = []
        kept_timestamps = []
        for timestamp in self.timestamps:
            descriptors = self.descriptors_from_features(feature_loader(timestamp))
            if descriptors.size == 0:
                continue
            kept_timestamps.append(timestamp)
            histograms.append(self._histogram(descriptors))

        self.timestamps = kept_timestamps
        self.histograms = np.stack(histograms)

        document_frequency = np.sum(self.histograms > 0, axis=0)
        self.idf = np.log((1.0 + len(self.histograms)) / (1.0 + document_frequency)) + 1.0
        self.histograms = self.histograms * self.idf[None, :]
        self.histograms = self._l2_normalize(self.histograms)

    def save(self, path: str) -> None:
        if self.vocab is None or self.histograms is None or self.idf is None:
            raise RuntimeError("Cannot save an unbuilt SuperPoint BoW index")
        np.savez_compressed(
            path,
            vocab=self.vocab,
            histograms=self.histograms,
            idf=self.idf,
            timestamps=np.asarray(self.timestamps, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str) -> "SuperPointBoWRetriever":
        data = np.load(path, allow_pickle=False)
        retriever = cls(vocab_size=int(data["vocab"].shape[0]))
        retriever.vocab = np.asarray(data["vocab"], dtype=np.float32)
        retriever.histograms = np.asarray(data["histograms"], dtype=np.float32)
        retriever.idf = np.asarray(data["idf"], dtype=np.float32)
        retriever.timestamps = [int(timestamp) for timestamp in data["timestamps"]]
        return retriever

    def _assign_words(self, descriptors: np.ndarray) -> np.ndarray:
        if self.vocab is None:
            raise RuntimeError("SuperPoint BoW retriever has not been built")
        words = []
        chunk_size = 2048
        for start in range(0, len(descriptors), chunk_size):
            chunk = descriptors[start:start + chunk_size]
            distances = (
                np.sum(chunk * chunk, axis=1, keepdims=True)
                - 2.0 * chunk @ self.vocab.T
                + np.sum(self.vocab * self.vocab, axis=1)[None, :]
            )
            words.append(np.argmin(distances, axis=1))
        return np.concatenate(words, axis=0) if words else np.empty((0,), dtype=np.int64)

    def _histogram(self, descriptors: np.ndarray) -> np.ndarray:
        if self.vocab is None or descriptors.size == 0:
            return np.zeros((0,), dtype=np.float32)
        words = self._assign_words(descriptors)
        hist = np.bincount(words, minlength=len(self.vocab)).astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        return hist

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)

    def query(self, features: dict, top_k: int) -> list[tuple[int, float]]:
        if self.histograms is None or self.vocab is None:
            return []
        descriptors = self.descriptors_from_features(features)
        if descriptors.size == 0:
            return []
        query_hist = self._histogram(descriptors)
        if self.idf is not None:
            query_hist = query_hist * self.idf
        query_hist = self._l2_normalize(query_hist[None, :])[0]
        scores = self.histograms @ query_hist
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices]
