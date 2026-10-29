"""Bag dropout and USB evidence must use the same host clock, not camera epochs."""
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest
from unittest.mock import patch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tool'))
from check_camera_bag import (
    correlate_usb, header_stamp, inspect_bag, main, read_kernel, resolve_bag,
)

START = 1_700_000_000_000_000_000


def cdr(stamp, little=True):
    sec, ns = divmod(stamp, 1_000_000_000)
    return bytes([0, int(little), 0, 0]) + struct.pack("<iI" if little else ">iI", sec, ns)


def fixture(directory, front=(0, .1, .2, .3), rear=(0, .1, .2, .3),
            omit_front_ir=(), tf=True):
    path = Path(directory)
    path.mkdir(exist_ok=True)
    (path / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
    cameras = [{"id": c, "prefix": p, "enabled": True}
               for c, p in (("front", "/camera/camera"), ("rear", "/camera/camera1"))]
    (path / "cameras.json").write_text(json.dumps({"cameras": cameras}))
    with sqlite3.connect(path / "bag_0.db3") as db:
        db.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT)")
        db.execute("CREATE TABLE messages (topic_id INTEGER, timestamp INTEGER, data BLOB)")
        tid = 0
        for camera, times in zip(cameras, (front, rear)):
            for suffix in ("infra1/image_rect_raw", "depth/image_rect_raw", "vio_image"):
                tid += 1
                db.execute("INSERT INTO topics VALUES (?,?)",
                           (tid, camera["prefix"] + "/" + suffix))
                for seconds in times:
                    if (camera["id"] == "front" and suffix.startswith("infra1/")
                            and seconds in omit_front_ir):
                        continue
                    ns = round(seconds * 1e9)
                    epoch = (100 if camera["id"] == "front" else 10000) * 10**9
                    db.execute("INSERT INTO messages VALUES (?,?,?)",
                               (tid, START + ns, cdr(epoch + ns)))
        tid += 1
        db.execute("INSERT INTO topics VALUES (?,?)", (tid, "/tf_static"))
        if tf:
            db.execute("INSERT INTO messages VALUES (?,?,?)", (tid, START, b"tf"))
    return path


def kernel(*rows):
    return {"status": "supplied", "source": "test", "note": "",
            "text": "\n".join(json.dumps({
                "__REALTIME_TIMESTAMP": str((START + round(seconds * 1e9)) // 1000),
                "MESSAGE": message,
            }) for seconds, message in rows)}


class CameraBagCheckTest(unittest.TestCase):
    def test_plain_cdr_endianness_and_malformed_headers(self):
        stamp = 101_123_456_789
        self.assertEqual(header_stamp(cdr(stamp)), stamp)
        self.assertEqual(header_stamp(cdr(stamp, False)), stamp)
        self.assertIsNone(header_stamp(b"short"))
        self.assertIsNone(header_stamp(bytes([0, 1, 0, 0]) + struct.pack("<iI", 1, 10**9)))

    def test_good_dual_bag_and_no_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(directory)
            before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in path.iterdir()}
            report = inspect_bag(path)
            self.assertEqual(report["warnings"], [])
            self.assertEqual(report["cameras"]["front"]["received_depth_pair_pct"], 100)
            self.assertEqual(report["cameras"]["rear"]["received_depth_pair_pct"], 100)
            self.assertEqual(report["tf_static_count"], 1)
            self.assertEqual(before, {
                p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in path.iterdir()})

    def test_whole_stream_dropout_is_not_hidden_by_high_pair_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(directory, front=(0, .1, 8.1, 8.2),
                           rear=(0, .1, .2, 4, 8, 8.2))
            report = inspect_bag(path)
            self.assertEqual(report["cameras"]["front"]["received_depth_pair_pct"], 100)
            gaps = report["topics"]["/camera/camera/vio_image"]["gaps"]
            self.assertEqual(gaps[0]["duration_s"], 8)
            self.assertTrue(any("missing interval" in w for w in report["warnings"]))
            logs = kernel((.2, "cdc_ncm 2-1.2:1.0 enx020000000a01: unregister 'cdc_ncm'"),
                          (.3, "usb 2-1.2: USB disconnect, device number 8"),
                          (8, "usb 2-1.2: new SuperSpeed USB device number 9"))
            result = correlate_usb(report, logs)
            self.assertEqual(len(result["gap_correlations"]), 3)
            self.assertTrue(all(m["camera"] == "front" for m in result["gap_correlations"]))

    def test_unrelated_rear_reset_does_not_prove_front_usb_dropout(self):
        with tempfile.TemporaryDirectory() as directory:
            report = inspect_bag(fixture(directory, front=(0, .1, 8.1),
                                         rear=(0, .1, 8.1)))
            result = correlate_usb(report, kernel(
                (.2, "cdc_ncm 2-1.1:1.0 enx020000000b01: unregister 'cdc_ncm'")))
            self.assertTrue(result["gap_correlations"])
            self.assertTrue(all(m["camera"] == "rear" for m in result["gap_correlations"]))

    def test_missing_tail_is_correlated(self):
        with tempfile.TemporaryDirectory() as directory:
            report = inspect_bag(fixture(directory, front=(0, .1), rear=(0, .1, 8.1)))
            result = correlate_usb(report, kernel(
                (.3, "cdc_ncm 2-1.2:1.0 enx020000000a01: unregister 'cdc_ncm'")))
            self.assertTrue(all(m["gap"]["boundary"] == "trailing"
                                for m in result["gap_correlations"]))

    def test_reduced_ir_pairing_and_missing_static_tf(self):
        with tempfile.TemporaryDirectory() as directory:
            report = inspect_bag(fixture(directory, omit_front_ir=(.1,), tf=False))
            self.assertEqual(report["cameras"]["front"]["received_depth_pair_pct"], 75)
            self.assertTrue(any("pairing below" in w for w in report["warnings"]))
            self.assertTrue(any("/tf_static" in w for w in report["warnings"]))

    def test_empty_topic_does_not_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            report = inspect_bag(fixture(directory, front=(), rear=(0, .1)))
            self.assertEqual(report["cameras"]["front"]["paired"], 0)
            self.assertTrue(any("missing required" in w for w in report["warnings"]))

    def test_split_bags_combine_topic_ids_by_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "a", front=(0, .1), rear=(0, .1))
            fixture(root / "b", front=(.2, .3), rear=(.2, .3))
            (root / "a/bag_0.db3").rename(root / "bag_0.db3")
            (root / "b/bag_0.db3").rename(root / "bag_1.db3")
            (root / "a/cameras.json").rename(root / "cameras.json")
            report = inspect_bag(root)
            self.assertEqual(len(report["sqlite_integrity"]), 2)
            self.assertEqual(report["cameras"]["front"]["paired"], 4)

    def test_missing_path_does_not_create_database(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.db3"
            with self.assertRaises(sqlite3.Error):
                inspect_bag(path)
            self.assertFalse(path.exists())
            with self.assertRaises(ValueError):
                resolve_bag("missing", Path(directory))

    def test_journal_permission_failure_is_unavailable(self):
        with patch("check_camera_bag.subprocess.run",
                   return_value=type("Result", (), {
                       "returncode": 1, "stdout": "", "stderr": "Permission denied"})()):
            result = read_kernel(START, START + 10**9)
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("Permission denied", result["note"])

    def test_journal_command_covers_previous_boots_and_no_records_is_uncertain(self):
        with patch("check_camera_bag.subprocess.run",
                   return_value=type("Result", (), {
                       "returncode": 0, "stdout": "", "stderr": ""})()) as run:
            result = read_kernel(START, START + 10**9)
        command = run.call_args.args[0]
        self.assertIn("_TRANSPORT=kernel", command)
        self.assertNotIn("-k", command)
        self.assertNotIn("-b", command)
        self.assertEqual(result["status"], "no_records")

    def test_cli_json_and_skip_usb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = fixture(root / "bag")
            output = root / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = main([str(path), "--skip-usb", "--json", str(output)])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(output.read_text())["usb"]["status"], "skipped")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = main([str(path), "--skip-usb", "--json", str(path/"metadata.yaml")])
            self.assertEqual(rc, 2)
            self.assertTrue((path/"metadata.yaml").read_text().startswith("rosbag2"))


if __name__ == "__main__":
    unittest.main()
