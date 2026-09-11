#!/usr/bin/env python3
"""Procedural textures for scripted sim scenes -- no binary assets in the repo.

Generates models/textures/FLOOR_Albedo.png plus crate_0..7.png (crate variants;
neighboring crates get different textures, see scene_runner.py) next to this
script. Deterministic by default (seed 42, override with TINYNAV_TEX_SEED);
run with --force to regenerate. Called by run_simulator.sh before gz starts
(the world SDF references the floor texture) and by scene_runner.py on startup
(gz caches textures by path, so after --force the gz server needs a restart).
"""

import os
import pathlib

import cv2
import numpy as np

TEX_DIR = pathlib.Path(__file__).resolve().parent / "models" / "textures"
SIZE = 1024


def noise(rng, cells, amp):
    g = rng.random((cells, cells)).astype(np.float32)
    return (cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC) - 0.5) * 2 * amp


def crate_texture(rng):
    """Stacked-cardboard look: shaded patches, dark seams, light labels."""
    img = np.zeros((SIZE, SIZE, 3), np.float32)
    img[:] = (0.55, 0.63, 0.72)  # warm cardboard base (BGR)
    n = noise(rng, 8, 0.10) + noise(rng, 64, 0.04) + noise(rng, 512, 0.02)
    img += n[..., None]
    grid = 4
    cell = SIZE // grid
    for gy in range(grid):
        for gx in range(grid):
            # per-patch shade/hue jitter, inset from cell edges
            m = int(rng.integers(6, 22))
            x0, y0 = gx * cell + m, gy * cell + m
            x1, y1 = (gx + 1) * cell - m, (gy + 1) * cell - m
            shade = rng.uniform(0.85, 1.2)
            tint = np.array([rng.uniform(0.92, 1.0), rng.uniform(0.95, 1.02), rng.uniform(0.98, 1.08)])
            img[y0:y1, x0:x1] *= shade * tint
            cv2.rectangle(img, (x0, y0), (x1, y1), (0.25, 0.28, 0.32), 3)  # seam
    for _ in range(4):  # shipping labels: light patch + dark stripe text
        w, h = int(rng.integers(60, 120)), int(rng.integers(40, 80))
        x, y = int(rng.integers(0, SIZE - w)), int(rng.integers(0, SIZE - h))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0.88, 0.88, 0.9), -1)
        for i in range(3):
            ly = y + 8 + i * (h - 16) // 3
            cv2.line(img, (x + 6, ly), (x + w - 6 - int(rng.integers(0, w // 3)), ly), (0.15, 0.15, 0.15), 2)
    return np.clip(img, 0, 1)


def floor_texture(rng):
    """Concrete: mid-gray base, large stains, fine grain, panel seams, scuffs."""
    img = np.zeros((SIZE, SIZE, 3), np.float32)
    img[:] = 0.45
    n = noise(rng, 6, 0.08) + noise(rng, 48, 0.04) + noise(rng, 512, 0.02)
    img += n[..., None]
    step = SIZE // 8
    for i in range(1, 8):  # panel seams with slight jitter
        j = int(rng.integers(-6, 6))
        cv2.line(img, (i * step + j, 0), (i * step + j, SIZE), (0.28, 0.28, 0.28), 2)
        cv2.line(img, (0, i * step + j), (SIZE, i * step + j), (0.28, 0.28, 0.28), 2)
    for _ in range(12):  # dark scuff ellipses
        x, y = int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE))
        axes = (int(rng.integers(20, 90)), int(rng.integers(8, 30)))
        cv2.ellipse(img, (x, y), axes, float(rng.uniform(0, 180)), 0, 360, (0.32, 0.32, 0.32), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return np.clip(img, 0, 1)


CRATE_PREFIX = "crate"
CRATE_VARIANTS = 8  # adjacent crates must differ: one repeated texture lets
                    # SLAM alias by one crate spacing (~1.3m x-drift observed)
FLOOR_FILE = "FLOOR_Albedo.png"


def texture_files():
    return [FLOOR_FILE] + [f"{CRATE_PREFIX}_{i}.png" for i in range(CRATE_VARIANTS)]


def ensure_textures(force=False, seed=None):
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    if all((TEX_DIR / f).exists() for f in texture_files()) and not force:
        return
    seed = int(os.environ.get("TINYNAV_TEX_SEED", "42")) if seed is None else seed
    rng = np.random.default_rng(seed)
    jobs = [(FLOOR_FILE, floor_texture)] + [
        (f"{CRATE_PREFIX}_{i}.png", crate_texture) for i in range(CRATE_VARIANTS)
    ]
    for fname, gen in jobs:
        img = (gen(rng) * 255).astype(np.uint8)
        cv2.imwrite(str(TEX_DIR / fname), img)
        print(f"generated {fname} (mean {img.mean():.0f}, seed {seed})")


if __name__ == "__main__":
    ensure_textures(force="--force" in os.sys.argv)
