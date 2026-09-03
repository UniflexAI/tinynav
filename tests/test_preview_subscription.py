"""Preview subscription lifecycle under concurrent viewers.

add_preview_callback / remove_preview_callback decide whether to create or
destroy the ROS subscription while holding the lock, but act on that decision
after releasing it. Two viewers churning on the same topic can therefore both
decide "I am the first" and both create, leaving a subscription that nothing
holds a reference to and that no later remove can destroy -- a duplicate reader
on an image topic, for the life of the process.

Usage:
    cd /tinynav
    python tests/test_preview_subscription.py
"""
from __future__ import annotations

import os
import sys
import threading
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.backend.node_manager import BackendNode

TOPIC = '/camera/camera/infra1/image_rect_raw'


class _Recorder:
    """Stands in for the rclpy subscription factory and records lifetimes."""

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
    node.preview_callbacks = {TOPIC: []}
    node._image_subs = {}
    node.create_subscription = rec.create_subscription
    node.destroy_subscription = rec.destroy_subscription
    return node


def test_churn_leaves_at_most_one_reader():
    rec = _Recorder()
    node = _bare_node(rec)

    # Hold the first viewer inside create_subscription so the second viewer's
    # add/remove/add cycle interleaves with it, the way two browser tabs do.
    gate = threading.Event()
    rec.block_next = gate
    first = threading.Thread(target=node.add_preview_callback, args=(TOPIC, 'cb1'))
    first.start()
    assert rec.entered.wait(5.0), 'the first viewer never reached create_subscription'

    node.remove_preview_callback(TOPIC, 'cb1')
    node.add_preview_callback(TOPIC, 'cb2')
    gate.set()
    first.join(5.0)
    assert not first.is_alive(), 'the first viewer never returned'

    live = rec.live()
    assert len(live) <= 1, (
        f'{len(live)} readers left on {TOPIC}; a duplicate subscription keeps '
        f'consuming the stream and no remove_preview_callback can reach it')
    for sub in live:
        assert sub is node._image_subs.get(TOPIC), (
            'a live subscription is not the one _image_subs holds, so nothing '
            'can ever destroy it')


def test_last_viewer_leaves_no_reader():
    rec = _Recorder()
    node = _bare_node(rec)
    node.add_preview_callback(TOPIC, 'cb1')
    node.add_preview_callback(TOPIC, 'cb2')
    node.remove_preview_callback(TOPIC, 'cb1')
    assert len(rec.live()) == 1, 'the stream must survive while a viewer remains'
    node.remove_preview_callback(TOPIC, 'cb2')
    assert rec.live() == [], 'the last viewer leaving must release the stream'


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
