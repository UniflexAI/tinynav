"""How much does the planning rule (single-frame camera depth) miss compared with the aggregated teacher label?

Counted inside a robot-relevant box (0.3-4 m ahead, |y| < 2 m), one-cell tolerance on obstacles.
  label quality : share of traversed cells the label calls obstacle (should be ~0, except stairs for step)
  miss / free   : label obstacles the baseline calls free (dangerous) or never saw
  false alarm   : label free cells the baseline calls obstacle
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .make_bev_gt import BEV_N, BEV_X, BEV_Y, load_frames
from .viz_bev import baseline, classes

res_b = (BEV_X[1] - BEV_X[0]) / BEV_N
xs = BEV_X[1] - (np.arange(BEV_N) + 0.5) * res_b
ys = BEV_Y[1] - (np.arange(BEV_N) + 0.5) * res_b
X, Y = np.meshgrid(xs, ys, indexing='ij')
ROI = (X > 0.3) & (X < 4.0) & (np.abs(Y) < 2.0)
KER = np.ones((3, 3), np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('frames_dirs', nargs='+')
    ap.add_argument('--bev', default='bev')
    ap.add_argument('--max_frames', type=int, default=0)
    args = ap.parse_args()
    rows = []
    for dd in args.frames_dirs:
        d = Path(dd)
        if not (d / args.bev / 'info.json').exists():
            continue
        meta, _, frames = load_frames(d)
        frames = frames[:args.max_frames] if args.max_frames else frames
        cam_h = json.loads((d / args.bev / 'info.json').read_text())['cam_h']
        c = dict(trav=0, trav_obs=0, trav_step=0, obs=0, obs_free=0, obs_unseen=0, step=0, step_free=0,
                 rail=0, rail_free=0, free=0, free_alarm=0)
        for f in frames:
            dp = d / f"{f['name']}_depth.png"
            if not dp.exists():
                continue
            lab = dict(np.load(d / args.bev / f"{f['name']}.npz"))
            cls = classes(lab)
            base = baseline(cv2.imread(str(dp), cv2.IMREAD_UNCHANGED), f['T'], meta['K'], cam_h)
            base_obs = cv2.dilate((base == 2).astype(np.uint8), KER).astype(bool)
            tr = lab['traversed'] & ROI
            c['trav'] += tr.sum()
            c['trav_obs'] += (tr & (cls == 2)).sum()
            c['trav_step'] += (tr & (cls == 4)).sum()
            for name, m in (('obs', cls == 2), ('step', cls == 4), ('rail', cls == 3)):
                m = m & ROI
                c[name] += m.sum()
                c[f'{name}_free'] += (m & (base == 1) & ~base_obs).sum()
                if name == 'obs':
                    c['obs_unseen'] += (m & (base == 0)).sum()
            fr = (cls == 1) & ROI & (base > 0)
            c['free'] += fr.sum()
            c['free_alarm'] += (fr & (base == 2)).sum()
        pct = lambda a, b: f'{100 * c[a] / max(c[b], 1):5.1f}%'
        rows.append((d.name, len(frames), pct('trav_obs', 'trav'), pct('trav_step', 'trav'), pct('obs_free', 'obs'),
                     pct('obs_unseen', 'obs'), pct('step_free', 'step'), pct('rail_free', 'rail'), pct('free_alarm', 'free')))
    hdr = ('bag', 'frames', 'trav->obs', 'trav->step', 'obs:base free', 'obs:base unseen', 'step:base free',
           'rail:base free', 'free:base obs')
    print(' | '.join(hdr))
    for r in rows:
        print(' | '.join(str(x) for x in r))


if __name__ == '__main__':
    main()
