"""Contact sheet of evenly spaced infra1 frames per bag, to see which scenes the data covers."""
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

TOPIC = '/camera/camera/infra1/image_rect_raw'
N = 8


def sample(bag):
    with AnyReader([bag], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as r:
        conns = [c for c in r.connections if c.topic == TOPIC]
        total = sum(c.msgcount for c in conns)
        picks = set(np.linspace(0, total - 1, N).astype(int))
        ims = []
        for i, (conn, t, raw) in enumerate(r.messages(connections=conns)):
            if i in picks:
                m = r.deserialize(raw, conn.msgtype)
                im = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
                im = cv2.resize(im, (im.shape[1] // 4, im.shape[0] // 4))
                cv2.putText(im, f'{(t - r.start_time) / 1e9:.0f}s', (3, 12), 0, 0.4, 255, 1)
                ims.append(im)
            if len(ims) == N:
                break
    h = max(i.shape[0] for i in ims)
    return np.hstack([cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in ims])


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for bag in sorted(p for p in root.iterdir() if (p / 'metadata.yaml').exists()):
        try:
            row = sample(bag)
        except Exception as e:  # skip broken/empty bags, keep going
            print(bag.name, 'skip:', e, flush=True)
            continue
        bar = np.zeros((18, row.shape[1]), np.uint8)
        cv2.putText(bar, bag.name, (3, 13), 0, 0.45, 255, 1)
        cv2.imwrite(str(out / f'{bag.name}.png'), np.vstack([bar, row]))
        print(bag.name, 'ok', flush=True)


if __name__ == '__main__':
    main()
