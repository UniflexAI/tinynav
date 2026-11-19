"""FlexGlue v0.1 sparse TensorRT matcher for existing SuperPoint keypoints."""
import hashlib
import os
import platform
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'
FEATURE_KEY = 'flexglue_v0_1'


class FlexGlueTRT:
    def __init__(self, model_dir=None):
        from tinynav.core.flexglue_trt import Session
        root = Path(model_dir or os.environ.get('TINYNAV_FLEXGLUE_MODEL_DIR', MODEL_DIR))
        arch = platform.machine()
        self.descriptor = Session(root / f'flexglue_descriptor_{arch}.plan')
        self.matcher = Session(root / f'flexglue_matcher_prescale_cuda_{arch}.plan')
        self._cache = OrderedDict()

    def describe(self, image):
        if image is None:
            raise ValueError('FlexGlue needs the infrared image or saved FlexGlue features')
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise ValueError('FlexGlue expects a uint8 grayscale or BGR image')
        key = (image.shape, hashlib.sha256(image.tobytes()).digest())
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if image.ndim == 2:
            rgb = np.repeat(image[:, :, None], 3, axis=2)
        elif image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError('FlexGlue expects one grayscale plane or three BGR channels')
        h, w = self.descriptor.get_inputs()[0].shape[-2:]
        rgb = cv2.resize(rgb.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_LINEAR)
        features, = self.descriptor.run(['features'], {'image': np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])})
        value = {'features': features, 'image_size': image.shape[1::-1]}
        self._cache[key] = value
        if len(self._cache) > 16:
            self._cache.popitem(last=False)
        return value

    def add_features(self, features, image):
        # Persist with scratch keyframes: their video encoder cannot be read yet.
        # FP16 engine output is usually exact in FP16, which halves the map size; keep FP32 otherwise.
        value = self.describe(image)
        half = value['features'].astype(np.float16)
        if np.array_equal(half.astype(np.float32), value['features']):
            value = {**value, 'features': half}
        return {**features, FEATURE_KEY: value}

    def _description(self, features, image):
        if FEATURE_KEY in features:
            value = features[FEATURE_KEY]
            return {**value, 'features': np.asarray(value['features'], dtype=np.float32)}
        return self.describe(image() if callable(image) else image)

    def match_keypoints(self, feats0, feats1, image0=None, image1=None):
        points = [np.asarray(f['kpts'])[0] for f in (feats0, feats1)]
        indices = [np.flatnonzero(np.asarray(f['mask']).reshape(-1)) for f in (feats0, feats1)]
        if any(len(i) == 0 for i in indices):
            return points[0][:0], points[1][:0], np.empty((0, 2), dtype=np.int64)
        a, b = (self._description(f, im) for f, im in ((feats0, image0), (feats1, image1)))
        feeds = {
            'features0': a['features'], 'features1': b['features'],
            # ONNX uses pixel corners; SuperPoint/PnP use pixel centers.
            'kpts0': (points[0][indices[0]][None] + 0.5).astype(np.float32),
            'kpts1': (points[1][indices[1]][None] + 0.5).astype(np.float32),
            'image_size0': np.asarray([a['image_size']], dtype=np.float32),
            'image_size1': np.asarray([b['image_size']], dtype=np.float32),
            'filter_threshold': np.asarray(0.1, dtype=np.float32),
            'max_distance_px': np.asarray(3.0, dtype=np.float32),
            'min_covisibility': np.asarray(0.0, dtype=np.float32),
        }
        matches0, = self.matcher.run(['matches0'], feeds)
        valid = np.flatnonzero(matches0[0] >= 0)
        matches = np.column_stack((indices[0][valid], indices[1][matches0[0, valid]])).astype(np.int64)
        return points[0][matches[:, 0]], points[1][matches[:, 1]], matches
