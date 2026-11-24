"""How a map's keyframe entries are encoded on disk: depth as uint16 millimetres,
SuperPoint descriptors in FP16 when that is exact. NumPy only, so the offline tools
can read a map without loading the models."""
import numpy as np

#: Depth is stored in whole millimetres; 0 means no depth, as `disparity_to_depth` writes it.
DEPTH_SCALE = np.float32(1000.0)
_DEPTH_MAX_MM = np.iinfo(np.uint16).max


def encode_depth(depth: np.ndarray) -> np.ndarray:
    """Metres (float) to uint16 millimetres, half of float32's size. Non-finite, <= 0
    and out-of-range depths become 0. Exact for depth already on a mm grid (Looper's)."""
    mm = np.rint(np.asarray(depth, dtype=np.float64) * float(DEPTH_SCALE))
    mm = np.where(np.isfinite(mm) & (mm > 0) & (mm <= _DEPTH_MAX_MM), mm, 0)
    return mm.astype(np.uint16)


def decode_depth(stored: np.ndarray) -> np.ndarray:
    """A stored depth back to float32 metres; maps written before `encode_depth` hold
    float32 already and come back as they are."""
    stored = np.asarray(stored)
    if stored.dtype == np.uint16:
        # The division looper_bridge_node does, so its depth comes back bit-identical.
        return stored.astype(np.float32) / DEPTH_SCALE
    return stored


def half_descriptors(features: dict) -> dict:
    """`features` with SuperPoint's descriptors in FP16 when that is exact (the FP16
    engine's output is), else unchanged. LightGlueTRT copies them into its float32
    input, so readers need no change."""
    descps = features.get('descps')
    if descps is None or np.asarray(descps).dtype != np.float32:
        return features
    half = descps.astype(np.float16)
    if not np.array_equal(half.astype(np.float32), descps):
        return features
    return {**features, 'descps': half}
