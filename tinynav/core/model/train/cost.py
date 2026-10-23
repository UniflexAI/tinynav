"""Robot-specific traversability from the robot-agnostic BEV geometry (labels or model output).

Geometry layers (heights in meters):
  known : a surface was observed / predicted in the cell
  span  : vertical extent of occupied space above the support surface (walls, boxes, railings)
  step  : height jump to neighbouring cells (stairs, curbs, drop edges)
  rail  : railing / fence
The model predicts span and step as exceedance probabilities P(x >= t) at fixed thresholds, so each robot
reads the threshold it cares about and uncertain cells stay uncertain instead of regressing to flat ground.
"""
from dataclasses import dataclass

import numpy as np

UNKNOWN, FREE, OBSTACLE = 0, 1, 2
SPAN_T = (0.10, 0.15, 0.25, 0.40, 0.60)
STEP_T = (0.06, 0.10, 0.15, 0.22, 0.30)
# network output channels: name -> count (known, ground and rail are single maps, span/step one per threshold)
HEADS = dict(known=1, ground=1, span=len(SPAN_T), step=len(STEP_T), rail=1)


def split_channels(out):
    """(..., C, N, N) array/tensor -> dict; single-channel heads drop their channel axis."""
    res, i = {}, 0
    for k, n in HEADS.items():
        res[k] = out[..., i, :, :] if n == 1 else out[..., i:i + n, :, :]
        i += n
    return res


@dataclass
class TraversalParams:
    max_step: float   # higher jumps to a neighbour cell are not traversable (curb, drop, riser); one of STEP_T
    max_span: float   # taller occupied extent in a cell is an obstacle; one of SPAN_T


PRESETS = {
    # stairs (risers ~0.15-0.18 m) are traversable, walls / drops / railings are not
    'quadruped': TraversalParams(max_step=0.22, max_span=0.25),
    # curbs and single steps already block a wheeled base
    'wheeled': TraversalParams(max_step=0.06, max_span=0.10),
}


def classify(known, span, step, rail, params):
    """Classes from metric geometry (labels)."""
    cls = np.full(np.shape(known), UNKNOWN, np.uint8)
    cls[known] = FREE
    cls[known & ((span >= params.max_span) | (step >= params.max_step) | rail)] = OBSTACLE
    return cls


def obstacle_prob(span_p, step_p, rail_p, params):
    """P(obstacle) from model probabilities; span_p (len(SPAN_T),H,W), step_p (len(STEP_T),H,W), rail_p (H,W)."""
    s = span_p[SPAN_T.index(params.max_span)]
    t = step_p[STEP_T.index(params.max_step)]
    return np.maximum(np.maximum(s, t), rail_p)


def classify_prob(known_p, span_p, step_p, rail_p, params, thr=0.5):
    """Classes from model probabilities. Obstacles count even where the model is unsure a surface exists."""
    obs = obstacle_prob(span_p, step_p, rail_p, params) >= thr
    cls = np.full(np.shape(known_p), UNKNOWN, np.uint8)
    cls[known_p >= 0.5] = FREE
    cls[obs] = OBSTACLE
    return cls
