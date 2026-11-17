"""Diagnostic requests cannot escape the bag directory or leak child processes."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import os
import sys

from fastapi import HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.backend.routers import bag_check


class FakeProcess:
    pid = 1234567

    def __init__(self, result_path, code=0, hold=False):
        self.returncode = None
        self.result_path = result_path
        self.code = code
        self.hold = hold

    async def communicate(self):
        if self.hold:
            await asyncio.Event().wait()
        self.returncode = self.code
        if self.code in (0, 1):
            self.result_path.write_text(json.dumps({"warnings": ["gap"] if self.code else []}))
        return b"diagnostic result", b"invalid database" if self.code == 2 else b""

    async def wait(self):
        self.returncode = -15
        return self.returncode


class BagCheckPathTest(unittest.TestCase):
    def test_only_direct_bag_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bags"
            root.mkdir()
            bag = root / "bag_ok"
            bag.mkdir()
            (bag / "bag_0.db3").touch()
            self.assertEqual(bag_check.checked_bag(root, "bag_ok"), bag)
            outside = Path(directory) / "outside"
            outside.mkdir()
            (root / "escape").symlink_to(outside, target_is_directory=True)
            for name in ("..", ".", "../outside", "escape", "bad;command"):
                with self.assertRaises(HTTPException) as raised:
                    bag_check.checked_bag(root, name)
                self.assertEqual(raised.exception.status_code, 400)
            with self.assertRaises(HTTPException) as raised:
                bag_check.checked_bag(root, "missing")
            self.assertEqual(raised.exception.status_code, 404)


class BagCheckProcessTest(unittest.IsolatedAsyncioTestCase):
    async def execute(self, code=0, hold=False, disconnected=False):
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=disconnected))
        processes = []

        async def spawn(*args, **kwargs):
            self.assertTrue(kwargs["start_new_session"])
            self.assertIn("--skip-usb", args)
            self.assertEqual(args[1], str(bag_check._TOOL))
            result = Path(args[args.index("--json") + 1])
            process = FakeProcess(result, code=code, hold=hold)
            processes.append(process)
            return process

        with patch.object(bag_check.asyncio, "create_subprocess_exec", side_effect=spawn), \
             patch.object(bag_check, "_POLL_SECONDS", .001), \
             patch.object(bag_check.os, "killpg") as kill:
            try:
                result = await bag_check.run_check(Path("/bags/bag_good"), request)
                return result, processes, kill.call_count
            except HTTPException as exc:
                return exc, processes, kill.call_count

    async def test_warnings_are_a_successful_report(self):
        result, processes, kills = await self.execute(code=1)
        self.assertEqual(result["report"]["warnings"], ["gap"])
        self.assertEqual(result["text"], "diagnostic result")
        self.assertEqual(kills, 0)
        self.assertFalse(processes[0].result_path.parent.exists())

    async def test_tool_input_error_is_not_reported_as_a_healthy_bag(self):
        result, _, _ = await self.execute(code=2)
        self.assertEqual(result.status_code, 422)

    async def test_cancel_terminates_group_and_releases_lock(self):
        result, processes, kills = await self.execute(hold=True, disconnected=True)
        self.assertEqual(result.status_code, 499)
        self.assertEqual(kills, 1)
        self.assertIsNotNone(processes[0].returncode)
        self.assertFalse(bag_check._lock.locked())
        self.assertFalse(processes[0].result_path.parent.exists())

    async def test_timeout_terminates_group(self):
        with patch.object(bag_check, "_TIMEOUT_SECONDS", 0):
            result, _, kills = await self.execute(hold=True)
        self.assertEqual(result.status_code, 504)
        self.assertEqual(kills, 1)

    async def test_concurrent_request_is_refused(self):
        async with bag_check._lock:
            with self.assertRaises(HTTPException) as raised:
                await bag_check.run_check(Path("/bags/other"), None)
        self.assertEqual(raised.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
