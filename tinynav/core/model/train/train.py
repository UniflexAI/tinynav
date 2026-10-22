"""Train BevNet on teacher BEV labels.

  python -m tinynav.core.model.train.pack <label_dir> <packed_dir>
  python -m tinynav.core.model.train.train --data <packed_dir> --out runs/bev_v1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .bev_geometry import cell_centers
from .cost import OBSTACLE, PRESETS, classify
from .dataset import BevDataset
from .model import BevNet, split_heads

VAL_BAGS = ['bag_2026_09_19_12_04_41', 'bag_2026_09_19_12_05_36', 'bag_1970_01_01_08_07_55',
            'bag_jd_back_2026_07_27_17_46_19', 'bag_lijialin_gt']
X, Y = cell_centers()
ROI = torch.from_numpy((X > 0.3) & (X < 4.0) & (np.abs(Y) < 2.0))
W = dict(known=1.0, ground=1.0, span=2.0, step=2.0, rail=1.0)


def losses(out, b):
    p = split_heads(out)
    k = b['known']
    n = k.sum().clamp(min=1)
    masked_l1 = lambda pred, tgt, clip: (F.smooth_l1_loss(pred, tgt.clamp(max=clip), reduction='none', beta=0.05) * k).sum() / n
    rail = F.binary_cross_entropy_with_logits(p['rail'], b['rail'], pos_weight=torch.tensor(4.0, device=k.device), reduction='none')
    return dict(
        known=F.binary_cross_entropy_with_logits(p['known'], k),
        ground=masked_l1(p['ground'], b['ground'], 3.0),
        span=masked_l1(p['span'], b['span'], 1.5),
        step=masked_l1(p['step'], b['step'], 1.0),
        rail=(rail * k).sum() / n,
    )


def to_classes(known, span, step, rail, preset):
    return torch.from_numpy(classify(known.numpy(), span.numpy(), step.numpy(), rail.numpy(), PRESETS[preset]))


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    stats = {p: dict(tp=0, fp=0, fn=0, tp_v=0, fn_v=0, tp_h=0, fn_h=0) for p in PRESETS}
    for b in loader:
        out = split_heads(model(*(b[k].to(dev, non_blocking=True) for k in ('img', 'raster', 'grid', 'valid'))).float().cpu())
        known = b['known'] > 0.5
        roi = known & ROI
        vis = b['visible'] > 0.5
        for preset in PRESETS:
            gt = to_classes(known, b['span'], b['step'], b['rail'] > 0.5, preset) == OBSTACLE
            pr = to_classes(torch.sigmoid(out['known']) > 0.5, out['span'], out['step'],
                            torch.sigmoid(out['rail']) > 0.5, preset) == OBSTACLE
            s = stats[preset]
            s['tp'] += (gt & pr & roi).sum().item()
            s['fp'] += (~gt & pr & roi).sum().item()
            s['fn'] += (gt & ~pr & roi).sum().item()
            s['tp_v'] += (gt & pr & roi & vis).sum().item()
            s['fn_v'] += (gt & ~pr & roi & vis).sum().item()
            s['tp_h'] += (gt & pr & roi & ~vis).sum().item()
            s['fn_h'] += (gt & ~pr & roi & ~vis).sum().item()
    res = {}
    for preset, s in stats.items():
        r = lambda tp, fn: tp / max(tp + fn, 1)
        res[preset] = dict(recall=r(s['tp'], s['fn']), precision=s['tp'] / max(s['tp'] + s['fp'], 1),
                           iou=s['tp'] / max(s['tp'] + s['fp'] + s['fn'], 1),
                           recall_seen=r(s['tp_v'], s['fn_v']), recall_unseen=r(s['tp_h'], s['fn_h']))
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--val_bags', nargs='*', default=VAL_BAGS)
    ap.add_argument('--max_steps', type=int, default=0, help='stop early (smoke test)')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_bags = sorted(p.name for p in Path(args.data).iterdir() if p.is_dir() and any(p.glob('*.npz')))
    train_bags = [b for b in all_bags if b not in args.val_bags]
    val_bags = [b for b in all_bags if b in args.val_bags]
    tr = BevDataset(args.data, train_bags, augment=True)
    va = BevDataset(args.data, val_bags)
    print(f'train {len(tr)} samples / {len(train_bags)} bags, val {len(va)} / {val_bags}', flush=True)
    tl = DataLoader(tr, args.bs, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True, persistent_workers=True)
    vl = DataLoader(va, args.bs, num_workers=args.workers, pin_memory=True)

    model = BevNet().to(dev)
    enc = list(model.encoder.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    opt = torch.optim.AdamW([dict(params=enc, lr=args.lr * 0.3), dict(params=rest)], lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr * 0.3, args.lr], total_steps=total, pct_start=0.05)
    amp = torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == 'cuda')
    step, log = 0, []
    for ep in range(args.epochs):
        t0, acc = time.time(), {}
        for b in tl:
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            with amp:
                o = model(b['img'], b['raster'], b['grid'], b['valid'])
            ls = losses(o.float(), b)
            loss = sum(W[k] * v for k, v in ls.items())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            for k, v in ls.items():
                acc[k] = acc.get(k, 0) + v.item()
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        n = max(1, min(len(tl), step - ep * len(tl)))
        msg = dict(epoch=ep, sec=round(time.time() - t0), **{k: round(v / n, 4) for k, v in acc.items()})
        if len(va):
            msg['val'] = evaluate(model, vl, dev)
        log.append(msg)
        print(json.dumps(msg), flush=True)
        torch.save(dict(model=model.state_dict(), epoch=ep, args=vars(args)), out / 'last.pt')
        (out / 'log.json').write_text(json.dumps(log, indent=1))
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == '__main__':
    main()
