"""Optional FlexGlue v0.1 sparse ONNX matcher for existing SuperPoint keypoints."""
from collections import OrderedDict
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np

MODEL_FILES = {
    'flexglue_descriptor.onnx': '93fab8a4ccde1b7331640915357d96cff0929ea87134872f37f7b0150d9914ff',
    'flexglue_matcher_fp16.onnx': 'f81cb68f27367171728c2c51ea5b60acab348520bc12154d4a3befa79f2dfe11',
}
MODEL_URL = 'https://github.com/cvg/FlexGlue/releases/download/v0.1/'
MODEL_DIR = Path(__file__).resolve().parents[1] / 'models' / 'flexglue'
FEATURE_KEY = 'flexglue_v0_1'


class FlexGlueONNX:
    def __init__(self, model_dir=None):
        import onnxruntime as ort

        root = Path(model_dir or os.environ.get('TINYNAV_FLEXGLUE_MODEL_DIR', MODEL_DIR))
        for name in MODEL_FILES:
            if not (root / name).is_file():
                raise FileNotFoundError(f'{root / name}: run python -m tool.download_flexglue')
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('FlexGlue requires an ONNX Runtime build with CUDA support')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        providers = [('CUDAExecutionProvider', {'arena_extend_strategy': 'kSameAsRequested'}), 'CPUExecutionProvider']
        self.descriptor = ort.InferenceSession(str(root / 'flexglue_descriptor.onnx'), sess_options=options, providers=providers)
        self.matcher = ort.InferenceSession(str(root / 'flexglue_matcher_fp16.onnx'), sess_options=options, providers=providers)
        if any(s.get_providers()[0] != 'CUDAExecutionProvider' for s in (self.descriptor, self.matcher)):
            raise RuntimeError('FlexGlue could not initialize the CUDA execution provider')
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
        return {**features, FEATURE_KEY: self.describe(image)}

    def _description(self, features, image):
        if FEATURE_KEY in features:
            return features[FEATURE_KEY]
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
