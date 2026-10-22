"""Compare the planning rule, BevNet and their union against the teacher labels on held-out bags.

  python -m tinynav.core.model.train.evaluate --ckpt runs/bev_v1/last.pt --data <packed_dir> --out runs/bev_v1/eval

Counted in the box 0.3-4 m ahead, |y| < 2 m. Per robot preset, per bag:
  miss         label obstacle the method does not mark (one-cell tolerance), split by cause and by
               seen-now (visible in the current frame) vs filled-in cells
  false alarm  label free cell the method marks as obstacle
Writes metrics.json and example panels (image | planning rule | model | label).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .bev_geometry import BEV_N, cam_to_bev_rotation, cell_centers, depth_to_points, to_grid
from .cost import FREE, OBSTACLE, PRESETS, classify, classify_prob
from .dataset import BevDataset
from .model import BevNet, split_heads
from .train import VAL_BAGS

X, Y = cell_centers()
ROI = (X > 0.3) & (X < 4.0) & (np.abs(Y) < 2.0)
KER = np.ones((3, 3), np.uint8)
PLAN_BAND, PLAN_SPAN = (-0.4, 0.4), 0.2   # planning_node ObstacleConfig (band rel. camera, min_wall_span)
COLORS = np.array([[90, 90, 90], [60, 170, 90], [220, 50, 40]], np.uint8)


def planning_rule(depth_m, Kd, R_cb):
    """planning_node.build_obstacle_map on one frame: z-span inside the camera-relative band."""
    pb = depth_to_points(depth_m, Kd, step=1, zmax=4.5) @ R_cb.T
    r, q, ok = to_grid(pb)
    seen = np.zeros((BEV_N, BEV_N), bool)
    seen[r[ok], q[ok]] = True
    sel = ok & (pb[:, 2] >= PLAN_BAND[0]) & (pb[:, 2] <= PLAN_BAND[1])
    hi = np.full((BEV_N, BEV_N), -9.0)
    lo = np.full((BEV_N, BEV_N), 9.0)
    np.maximum.at(hi, (r[sel], q[sel]), pb[sel, 2])
    np.minimum.at(lo, (r[sel], q[sel]), pb[sel, 2])
    cls = np.where(seen, FREE, 0).astype(np.uint8)
    cls[(hi - lo) >= PLAN_SPAN] = OBSTACLE
    return cls


def label_causes(z, p):
    known = z['known'].astype(bool)
    span, step, rail = z['span'].astype(np.float32), z['step'].astype(np.float32), z['rail'].astype(bool)
    cls = classify(known, span, step, rail, p)
    obs = cls == OBSTACLE
    return cls, dict(rail=obs & rail, wall=obs & ~rail & (span >= p.max_span),
                     step=obs & ~rail & (span < p.max_span) & (step >= p.max_step))


def render(cls, scale=3):
    img = COLORS[cls]
    img = cv2.resize(img, (BEV_N * scale, BEV_N * scale), interpolation=cv2.INTER_NEAREST)
    cv2.circle(img, (BEV_N * scale // 2, int(BEV_N * scale * 4.8 / 6.4)), 5, (255, 255, 0), -1)
    return img


def new_counts():
    c = {}
    for m in ('rule', 'model', 'union'):
        for k in ('wall', 'rail', 'step', 'seen', 'unseen', 'all'):
            c[f'{m}.{k}.n'] = c[f'{m}.{k}.miss'] = 0
        c[f'{m}.free.n'] = c[f'{m}.free.alarm'] = 0
    return c


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bags', nargs='*', default=VAL_BAGS)
    ap.add_argument('--thr', type=float, default=0.5, help='obstacle probability threshold')
    ap.add_argument('--examples', type=int, default=4, help='panels per bag (frames where the rule misses most)')
    args = ap.parse_args()
    out = Path(args.out)
    (out / 'panels').mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BevNet(pretrained=False).to(dev).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)['model'])

    results = {}
    for bag in args.bags:
        ds = BevDataset(args.data, [bag])
        if not len(ds):
            continue
        counts = {p: new_counts() for p in PRESETS}
        ranked = []
        for i in range(len(ds)):
            s = ds[i]
            z = np.load(ds.files[i])
            o = split_heads(model(*(s[k][None].to(dev) for k in ('img', 'raster', 'grid', 'valid'))).float().cpu())
            o = {k: torch.sigmoid(v[0]).numpy() for k, v in o.items() if k != 'ground'}
            rule = planning_rule(z['depth'].astype(np.float32) / 1000.0, z['K_depth'], cam_to_bev_rotation(z['R_cw']))
            vis = z['visible'].astype(bool)
            for pname, p in PRESETS.items():
                lab, causes = label_causes(z, p)
                pred = classify_prob(o['known'], o['span'], o['step'], o['rail'], p, args.thr)
                methods = dict(rule=rule == OBSTACLE, model=pred == OBSTACLE)
                methods['union'] = methods['rule'] | methods['model']
                lab_obs = (lab == OBSTACLE) & ROI
                lab_free = (lab == FREE) & ROI
                c = counts[pname]
                for m, mo in methods.items():
                    hit = cv2.dilate(mo.astype(np.uint8), KER).astype(bool)
                    for k, mask in (('all', lab_obs), ('seen', lab_obs & vis), ('unseen', lab_obs & ~vis),
                                    *((k, v & ROI) for k, v in causes.items())):
                        c[f'{m}.{k}.n'] += int(mask.sum())
                        c[f'{m}.{k}.miss'] += int((mask & ~hit).sum())
                    c[f'{m}.free.n'] += int(lab_free.sum())
                    c[f'{m}.free.alarm'] += int((lab_free & mo).sum())
                if pname == 'quadruped':
                    miss_rule = int((lab_obs & ~cv2.dilate(methods['rule'].astype(np.uint8), KER).astype(bool)).sum())
                    ranked.append((miss_rule, i, rule, pred, lab))
        res = {}
        for pname, c in counts.items():
            res[pname] = {m: {k: round(c[f'{m}.{k}.miss'] / max(c[f'{m}.{k}.n'], 1), 4)
                              for k in ('all', 'seen', 'unseen', 'wall', 'rail', 'step')}
                          | {'false_alarm': round(c[f'{m}.free.alarm'] / max(c[f'{m}.free.n'], 1), 4),
                             'n_obstacle': c[f'{m}.all.n']}
                          for m in ('rule', 'model', 'union')}
        results[bag] = dict(frames=len(ds), **res)
        print(bag, json.dumps(res['quadruped']['rule']), json.dumps(res['quadruped']['model']), flush=True)

        # examples: spread over the bag, preferring frames where the rule misses most
        ranked.sort(key=lambda r: -r[0])
        picked = []
        for r in ranked:
            if len(picked) == args.examples:
                break
            if all(abs(r[1] - p[1]) > max(5, len(ds) // 20) for p in picked):
                picked.append(r)
        for _, i, rule, pred, lab in picked:
            img = (ds[i]['img'][0].numpy() * 255).astype(np.uint8)
            img = cv2.cvtColor(cv2.resize(img, (int(img.shape[1] * 384 / img.shape[0]), 384)), cv2.COLOR_GRAY2RGB)
            row = np.hstack([img] + [render(c) for c in (rule, pred, lab)])
            cv2.imwrite(str(out / 'panels' / f'{bag}_{ds.files[i].stem}.jpg'), row[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / 'metrics.json').write_text(json.dumps(results, indent=1))
    print('wrote', out / 'metrics.json')


if __name__ == '__main__':
    main()
