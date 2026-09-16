"""The voxel subscription must track the planning-stream viewers.

The cloud is decoded into Python points on every frame, so a subscription left
behind after the last viewer leaves costs more than anything else this node does
per frame -- and one created twice can never be freed, since only one of them is
named by _voxel_sub.

Usage:
    cd /tinynav
    python tests/test_planning_watcher.py
"""
from __future__ import annotations

import os
import sys
import threading
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.backend.node_manager import BackendNode


class _Recorder:
    def __init__(self):
        self.created = []
        self.destroyed = []
        self.block_next = None
        self.entered = threading.Event()

    def create_subscription(self, msg_type, topic, callback, qos):
        gate = self.block_next
        if gate is not None:
            self.block_next = None
            self.entered.set()
            gate.wait(5.0)
        sub = object()
        self.created.append(sub)
        return sub

    def destroy_subscription(self, sub):
        self.destroyed.append(sub)

    def live(self):
        return [s for s in self.created if s not in self.destroyed]


def _bare_node(rec):
    node = BackendNode.__new__(BackendNode)
    node._lock = threading.Lock()
    node._subs_lock = threading.Lock()
    node._planning_watchers = 0
    node._voxel_sub = None
    node._voxel_points = [{'x': 1.0, 'y': 2.0, 'z': 3.0}]
    node.create_subscription = rec.create_subscription
    node.destroy_subscription = rec.destroy_subscription
    return node


def test_no_viewer_no_subscription():
    rec = _Recorder()
    node = _bare_node(rec)
    node.add_planning_watcher()
    assert len(rec.live()) == 1, 'the first viewer must open the stream'
    node.add_planning_watcher()
    assert len(rec.live()) == 1, 'a second viewer must share the one subscription'
    node.remove_planning_watcher()
    assert len(rec.live()) == 1, 'the stream must survive while a viewer remains'
    node.remove_planning_watcher()
    assert rec.live() == [], 'the last viewer leaving must release the stream'
    assert node._voxel_points == [], 'stale points must not outlive the stream'


def test_churn_leaves_at_most_one_reader():
    rec = _Recorder()
    node = _bare_node(rec)
    gate = threading.Event()
    rec.block_next = gate
    first = threading.Thread(target=node.add_planning_watcher)
    first.start()
    assert rec.entered.wait(5.0), 'the first viewer never reached create_subscription'

    node.remove_planning_watcher()
    node.add_planning_watcher()
    gate.set()
    first.join(5.0)
    assert not first.is_alive(), 'the first viewer never returned'

    live = rec.live()
    assert len(live) <= 1, f'{len(live)} readers left; a duplicate can never be freed'
    for sub in live:
        assert sub is node._voxel_sub, 'a live subscription is not the one _voxel_sub names'


def main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith('test_'):
            continue
        try:
            fn()
            print(f'  PASS  {name}')
        except Exception:
            failures += 1
            print(f'  FAIL  {name}')
            traceback.print_exc()
    print('FAILED' if failures else 'OK')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
