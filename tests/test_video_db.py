"""tool/video_db.py's reader cache, decoder threads and GOP, and
tool/reencode_video_db.py."""
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from tool import video_db as V
from tool.reencode_video_db import backup_path, keyframe_gaps, main as reencode_main


def _frames(n, h=48, w=64):
    """Gray frames that differ from each other, as a map's IR keyframes do."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (h, w), dtype=np.uint8)
    return [np.roll(base, i, axis=1) for i in range(n)]


def _write(dir_path, frames, gop):
    with mock.patch.object(V, 'GOP', gop):
        db = V.VideoDB(dir_path, 'write')
        for i, f in enumerate(frames):
            db.write(1000 + i, f)
        db.close()


class GopTest(unittest.TestCase):
    def test_a_written_video_has_a_keyframe_every_gop_frames(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, _frames(40), 15)
            n, gap, _ = keyframe_gaps(os.path.join(d, 'video.mp4'))
        self.assertEqual(n, 40)
        self.assertLessEqual(gap, 15)

    def test_gop_0_leaves_it_to_x264(self):
        with tempfile.TemporaryDirectory() as d:
            _write(d, _frames(40), 0)
            _, gap, _ = keyframe_gaps(os.path.join(d, 'video.mp4'))
        self.assertGreater(gap, 15, 'x264 kept its own, longer GOP')


class ReaderTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.frames = _frames(30)
        _write(self.dir.name, self.frames, 15)

    def _reader(self):
        db = V.VideoDB(self.dir.name, 'read')
        self.addCleanup(db.close)
        return db

    def test_a_frame_read_again_is_not_decoded_again(self):
        db = self._reader()
        first = db.read(1007)
        self.assertEqual(db.decode_count, 1)
        again = db.read(1007)
        self.assertEqual(db.decode_count, 1, 'decoded twice')
        np.testing.assert_array_equal(first, again)
        self.assertLess(np.abs(first.astype(int) - self.frames[7]).mean(), 3.0)

    def test_each_read_is_its_own_copy(self):
        db = self._reader()
        first = db.read(1003)
        kept = first.copy()
        first[:] = 0
        np.testing.assert_array_equal(db.read(1003), kept)

    def test_the_cache_keeps_its_budget_dropping_the_least_recently_used(self):
        one = self.frames[0].nbytes
        with mock.patch.object(V, 'CACHE_MB', 2.5 * one / 1e6):
            db = self._reader()
            db.read(1000), db.read(1001), db.read(1000), db.read(1002)   # 1001 goes
            self.assertEqual(list(db._cache), [0, 2], 'by frame index, 1 dropped')
            self.assertLessEqual(db._cache_bytes, 2.5 * one)
            n = db.decode_count
            db.read(1000)
            self.assertEqual(db.decode_count, n, 'the recently used one was dropped')
            db.read(1001)
            self.assertEqual(db.decode_count, n + 1)

    def test_cache_0_keeps_nothing(self):
        with mock.patch.object(V, 'CACHE_MB', 0.0):
            db = self._reader()
            db.read(1004), db.read(1004)
        self.assertEqual((db.decode_count, len(db._cache)), (2, 0))

    def test_the_reader_decodes_on_the_threads_it_is_given(self):
        real = V.decord.VideoReader
        seen = {}

        def reader(path, **kw):
            seen.update(kw)
            return real(path, **kw)

        with mock.patch.object(V.decord, 'VideoReader', side_effect=reader), \
                mock.patch.object(V, 'DECODE_THREADS', 2):
            self._reader().read(1001)
        self.assertEqual(seen.get('num_threads'), 2)

    def test_an_unknown_timestamp_reads_nothing(self):
        self.assertIsNone(self._reader().read(5))


class ReencodeTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.map = os.path.join(self.root.name, 'maps', 'm')
        self.sub = os.path.join(self.map, 'infra1_images_db')
        self.frames = _frames(60)
        _write(self.sub, self.frames, 0)          # x264's own: one keyframe
        with open(os.path.join(self.sub, 'meta.json')) as f:
            self.meta = f.read()

    def test_it_reencodes_with_a_short_gop_keeping_frames_and_meta(self):
        self.assertEqual(reencode_main([self.map, '--gop', '15']), 0)
        video = os.path.join(self.sub, 'video.mp4')
        n, gap, _ = keyframe_gaps(video)
        self.assertEqual((n, gap <= 15), (60, True))
        with open(os.path.join(self.sub, 'meta.json')) as f:
            self.assertEqual(f.read(), self.meta, 'meta.json touched')
        backup = backup_path(self.map, 'infra1_images_db')
        self.assertEqual(backup, os.path.join(self.root.name, 'video-backup', 'm',
                                              'infra1_images_db', 'video.mp4'))
        self.assertGreater(keyframe_gaps(backup)[1], 15, 'the old one is kept as it was')
        db = V.VideoDB(self.sub, 'read')
        self.addCleanup(db.close)
        self.assertLess(np.abs(db.read(1042).astype(int) - self.frames[42]).mean(), 4.0)
        self.assertNotIn('infra1_images_db.reencode-tmp', os.listdir(self.map))

    def test_twice_is_refused_and_restore_puts_the_old_one_back(self):
        reencode_main([self.map])
        self.assertEqual(reencode_main([self.map]), 1, 'a second run over the backup')
        self.assertEqual(reencode_main([self.map, '--restore']), 0)
        self.assertGreater(keyframe_gaps(os.path.join(self.sub, 'video.mp4'))[1], 15)
        self.assertFalse(os.path.exists(backup_path(self.map, 'infra1_images_db')))

    def test_a_video_already_short_is_left(self):
        reencode_main([self.map])
        os.remove(backup_path(self.map, 'infra1_images_db'))
        self.assertEqual(reencode_main([self.map]), 0)
        self.assertFalse(os.path.exists(backup_path(self.map, 'infra1_images_db')))

    def test_a_meta_that_does_not_number_the_frames_is_not_touched(self):
        with open(os.path.join(self.sub, 'meta.json'), 'w') as f:
            json.dump({'ts_to_idx': {'1': 0, '2': 5}, 'is_gray': True}, f)
        self.assertEqual(reencode_main([self.map]), 1)
        self.assertGreater(keyframe_gaps(os.path.join(self.sub, 'video.mp4'))[1], 15)


if __name__ == '__main__':
    unittest.main()
