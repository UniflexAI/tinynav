"""Robot-specific traversability from the robot-agnostic BEV geometry (labels or model output).

Geometry layers (heights in meters):
  known : a surface was observed / predicted in the cell
  span  : vertical extent of occupied space above the support surface (walls, boxes, railings)
  step  : height jump to neighbouring cells (stairs, curbs, drop edges)
  rail  : railing / fence
"""
from dataclasses import dataclass

import numpy as np

UNKNOWN, FREE, OBSTACLE = 0, 1, 2


@dataclass
class TraversalParams:
    max_step: float   # higher jumps to a neighbour cell are not traversable (curb, drop, riser)
    max_span: float   # taller occupied extent in a cell is an obstacle


PRESETS = {
    # stairs (risers ~0.15-0.18 m) are traversable, walls / drops / railings are not
    'quadruped': TraversalParams(max_step=0.22, max_span=0.25),
    # curbs and single steps already block a wheeled base
    'wheeled': TraversalParams(max_step=0.06, max_span=0.10),
}


def classify(known, span, step, rail, params):
    cls = np.full(np.shape(known), UNKNOWN, np.uint8)
    cls[known] = FREE
    cls[known & ((span >= params.max_span) | (step >= params.max_step) | rail)] = OBSTACLE
    return cls
