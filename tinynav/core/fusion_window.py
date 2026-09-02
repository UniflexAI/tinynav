"""Which relocalization constraints the pose-graph solve should still believe.

Each constraint is one observation's implied map->odom -- true when it was taken, and
only while odom has not drifted since. Upstream bounds them by count (the newest 100),
which at ~0.8 relocalizations a second reaches back two minutes; this bounds them by
travel instead.

Measured on 118 standing still: 20 consecutive PnP answers agreed on (42.44, -4.47)
while the fused estimate sat 0.75m away, because a good observation was one vote
against ninety-nine older ones. After: 1cm apart.
"""
import os

import numpy as np

#: Metres of odom travel after which a constraint is stale. Travel and not time:
#: drift comes with travel, so standing still expires nothing -- which is what keeps a
#: stationary robot from ending up with no constraints at all.
FUSE_MAX_M = float(os.environ.get("TINYNAV_FUSE_MAX_M", "3.0"))
#: Never solve over fewer: one constraint is one observation, with no averaging left
#: to damp an aliased match.
FUSE_MIN = int(os.environ.get("TINYNAV_FUSE_MIN", "5"))
#: Upstream's cap, kept as the upper bound.
FUSE_MAX = int(os.environ.get("TINYNAV_FUSE_MAX", "100"))


def select_fusion_constraints(constraints, odom_poses):
    """The constraints taken within FUSE_MAX_M of driving of where we are now.

    "Now" is the newest constraint's own odom, not `latest_odom_pose`: the solve this
    runs inside was triggered by that observation. `odom_poses[i]` is the odom pose
    constraint `i` was observed at -- the only thing the solver has no way to ask
    about. Returns the constraints to solve over, newest last.
    """
    if len(constraints) <= FUSE_MIN:
        return constraints
    here = np.asarray(odom_poses[-1])[:3, 3]
    travel = np.linalg.norm(
        np.asarray([np.asarray(T)[:3, 3] for T in odom_poses]) - here, axis=1)
    fresh = [c for c, d in zip(constraints, travel) if d <= FUSE_MAX_M]
    if len(fresh) < FUSE_MIN:
        fresh = constraints[-FUSE_MIN:]
    return fresh[-FUSE_MAX:]
