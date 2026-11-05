import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tinynav.core.flexglue import FEATURE_KEY, FlexGlueONNX
from tool import download_flexglue


def features(points, valid):
    return {'kpts': np.asarray([points], dtype=np.float32),
            'mask': np.asarray(valid, dtype=bool)[None, :, None]}


class FlexGlueTest(unittest.TestCase):
    def setUp(self):
        self.model = FlexGlueONNX.__new__(FlexGlueONNX)
        from collections import OrderedDict
        self.model._cache = OrderedDict()
        self.model.descriptor = Mock()
        self.model.descriptor.get_inputs.return_value = [SimpleNamespace(shape=[1, 3, 8, 8])]
        self.model.descriptor.run.return_value = [np.zeros((1, 256, 1, 1), np.float32)]
        self.model.matcher = Mock()

    def test_indices_and_pixel_origins_survive_sparse_mask(self):
        a = features([[1, 2], [9, 9], [3, 4]], [1, 0, 1])
        b = features([[5, 6], [8, 8], [7, 8]], [1, 0, 1])
        self.model.matcher.run.return_value = [np.array([[1, -1]], np.int64)]
        p0, p1, matches = self.model.match_keypoints(a, b, np.zeros((10, 20), np.uint8), np.zeros((30, 40), np.uint8))
        np.testing.assert_array_equal(matches, [[0, 2]])
        np.testing.assert_array_equal(p0, [[1, 2]])
        np.testing.assert_array_equal(p1, [[7, 8]])
        feed = self.model.matcher.run.call_args.args[1]
        np.testing.assert_array_equal(feed['kpts0'], [[[1.5, 2.5], [3.5, 4.5]]])
        np.testing.assert_array_equal(feed['image_size0'], [[20, 10]])
        np.testing.assert_array_equal(feed['image_size1'], [[40, 30]])
        self.assertEqual(feed['min_covisibility'], 0)

    def test_empty_mask_does_not_decode_images_or_run_models(self):
        load = Mock(side_effect=AssertionError('unexpected decode'))
        a = features([[1, 2]], [0]); b = features([[3, 4]], [1])
        p0, p1, matches = self.model.match_keypoints(a, b, load, load)
        self.assertEqual(p0.shape, (0, 2)); self.assertEqual(p1.shape, (0, 2))
        self.assertEqual(matches.shape, (0, 2))
        self.model.matcher.run.assert_not_called(); load.assert_not_called()

    def test_no_matches_has_stable_two_column_shape(self):
        a = features([[1, 2]], [1]); self.model.matcher.run.return_value = [np.array([[-1]])]
        _, _, matches = self.model.match_keypoints(a, a, np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.uint8))
        self.assertEqual(matches.shape, (0, 2))

    def test_saved_features_work_without_reading_scratch_video(self):
        a = features([[1, 2]], [1]); image = np.zeros((10, 20), np.uint8)
        enriched = self.model.add_features(a, image)
        self.assertNotIn(FEATURE_KEY, a)
        self.model.matcher.run.return_value = [np.array([[0]])]
        load = Mock(side_effect=AssertionError('scratch video is write-only'))
        self.model.match_keypoints(enriched, enriched, load, load)
        load.assert_not_called(); self.assertEqual(self.model.descriptor.run.call_count, 1)

    def test_old_map_uses_lazy_image_and_descriptor_cache_is_bounded(self):
        a = features([[1, 2]], [1]); self.model.matcher.run.return_value = [np.array([[0]])]
        load = Mock(return_value=np.zeros((10, 20), np.uint8))
        self.model.match_keypoints(a, a, load, load)
        self.assertEqual(load.call_count, 2); self.assertEqual(self.model.descriptor.run.call_count, 1)
        feed = self.model.descriptor.run.call_args.args[1]['image']
        np.testing.assert_array_equal(feed[:, 0], feed[:, 1])
        for i in range(20): self.model.describe(np.full((10, 20), i, np.uint8))
        self.assertEqual(len(self.model._cache), 16)

    def test_missing_image_fails_instead_of_using_wrong_coordinates(self):
        a = features([[1, 2]], [1])
        with self.assertRaisesRegex(ValueError, 'infrared image'):
            self.model.match_keypoints(a, a)

    def test_constructor_rejects_cpu_only_runtime(self):
        ort = Mock(); ort.get_available_providers.return_value = ['CPUExecutionProvider']
        with tempfile.TemporaryDirectory() as directory:
            for name in download_flexglue.MODEL_FILES: (Path(directory) / name).touch()
            with patch.dict('sys.modules', {'onnxruntime': ort}):
                with self.assertRaisesRegex(RuntimeError, 'CUDA support'): FlexGlueONNX(directory)


class DownloadTest(unittest.TestCase):
    def test_bad_download_keeps_existing_file_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'model.onnx'; target.write_bytes(b'existing')
            def retrieve(url, path): Path(path).write_bytes(b'bad download')
            with patch.object(download_flexglue, 'MODEL_FILES', {'model.onnx': hashlib.sha256(b'correct').hexdigest()}), patch.object(download_flexglue.urllib.request, 'urlretrieve', retrieve):
                with self.assertRaisesRegex(ValueError, 'SHA256'): download_flexglue.download(directory)
            self.assertEqual(target.read_bytes(), b'existing')
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_verified_local_file_does_not_use_network(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'model.onnx'; target.write_bytes(b'correct')
            with patch.object(download_flexglue, 'MODEL_FILES', {'model.onnx': hashlib.sha256(b'correct').hexdigest()}), patch.object(download_flexglue.urllib.request, 'urlretrieve') as request:
                download_flexglue.download(directory); request.assert_not_called()


if __name__ == '__main__': unittest.main()
