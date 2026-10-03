"""Observed labels and actual captured depth, with no hidden scene map."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from tool.simulator.recovery_strategies import poses

p = argparse.ArgumentParser()
p.add_argument('--results', required=True)
a = p.parse_args()
root = Path(a.results)
r = json.loads((root / 'snapshot-1.json').read_text())
xy = r['branches']['scan']['anchor_xy']
yaw = r['branches']['scan']['anchor_yaw_deg']
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
colormap = ListedColormap(['#b8bec8', '#fafbfc', '#243b55'])
plan = next(plan for plan in r['fixed_candidates'] if plan['id'] == 'retreat_180_right')
trace = np.array([point for point, _ in poses(xy, yaw, plan['stages'])])
for ax, label, audit in zip(axes[0], ['Before scan', 'Stationary control', 'After 360-degree scan'],
                          [r['branches']['scan']['cells_before'], r['branches']['hold']['cells_after'], r['branches']['scan']['cells_after']]):
    cells = {(i, j): value for i, j, value, _ in audit['cells']}
    resolution = audit['resolution']
    i0, j0 = int(np.floor((xy[0]-3)/resolution)), int(np.floor((xy[1]-3)/resolution))
    raster = np.array([[{'clear': 1, 'blocked': 2}.get(cells.get((i0+i, j0+j)), 0) for i in range(60)] for j in range(60)])
    ax.imshow(raster, origin='lower', vmin=0, vmax=2, cmap=colormap,
              extent=[i0*resolution-xy[0], (i0+60)*resolution-xy[0], j0*resolution-xy[1], (j0+60)*resolution-xy[1]])
    ax.plot(trace[:, 0]-xy[0], trace[:, 1]-xy[1], color='#de6037', linewidth=2)
    ax.plot(0, 0, 'o', color='#126bc5')
    ax.arrow(0, 0, .45, 0, width=.035, color='#126bc5')
    ax.set_title(label)
    ax.set_xlabel('Forward (m)')
    ax.set_ylabel('Left (m)')
    ax.set_aspect('equal')
axes[0, 0].legend(handles=[Patch(color='#b8bec8', label='Unknown'), Patch(color='#fafbfc', label='Measured clear'),
                         Patch(color='#243b55', label='Measured blocked'), Patch(color='#de6037', label='Retreat candidate (not executed)')],
                  loc='lower left', fontsize=7)
scanned = np.load(root / 'snapshot-1-scan-depth.npz')
held = np.load(root / 'snapshot-1-hold-depth.npz')
for ax, label, depth in zip(axes[1], ['Forward depth during scan', 'Forward depth while stationary', 'Rear depth during scan'],
                          [scanned['0'], held['0'], scanned['2']]):
    image = ax.imshow(np.ma.masked_where(depth <= 0, depth), vmin=0, vmax=r['config']['camera']['max_range'], cmap='viridis')
    ax.set_title(label)
    ax.set_xlabel('Camera pixel u')
    ax.set_ylabel('Camera pixel v')
fig.colorbar(image, cax=fig.add_axes([.935, .14, .015, .25]), label='Actual simulated depth (m)')
fig.suptitle('Dead end: sensor-only observation, zero translation; no retreat executed', fontsize=13)
fig.subplots_adjust(top=.91, bottom=.11, left=.065, right=.90, wspace=.3, hspace=.35)
fig.text(.5, .025, 'Depth panels: white = no return (unknown). Candidate line was not executed.', ha='center', fontsize=9)
fig.savefig(root / 'observation-comparison.png', dpi=150)
print(root / 'observation-comparison.png')
