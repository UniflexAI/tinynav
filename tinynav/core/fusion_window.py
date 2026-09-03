"""Which relocalization constraints the pose-graph solve should still believe.

Each constraint is one observation's implied map->odom. Two things can be wrong with
one: it can be OLD -- true when it was taken, and only while odom has not drifted
since -- or it can be WRONG, an aliased match from somewhere the robot is not. This
answers both, and it is the only place either can be answered well.

**Stale, by travel.** Upstream bounds constraints by count (the newest 100), which at
~0.8 relocalizations a second reaches back two minutes. Measured on 118 standing
still: 20 consecutive PnP answers agreed on (42.44, -4.47) while the fused estimate
sat 0.75m away, because a good observation was one vote against ninety-nine older
ones. After bounding by travel: 1cm apart. Travel and not time, so standing still
expires nothing and a stationary robot is never starved.

**Wrong, by consensus.** The surviving constraints are then reduced to those lying on
the window's own regression line -- what the transform is doing, fitted robustly and
without reference to whatever the current estimate happens to be. This is the one place
the corridor aliasing can be answered: an aliased match is a minority report, and a
minority simply loses the vote. Nothing has to judge it, and nothing has to know which
of the two it is. Judging observations one at a time cannot do this, and two attempts
to were measured failing in opposite directions on 2026-09-03: refusing what disagreed
with the estimate defended a wrong pose through 157 consecutive refusals over 20m, 104
of them the truth; letting the rejected ones overrule it when enough agreed believed
ten aliased matches that agreed with each other to centimetres and moved the pose
49.65m.

**A line and not a cluster**, because the transform moves across a window: the map's
keyframe poses are themselves a VIO trajectory, so the map is locally consistent and
globally warped and the true transform slides as the robot travels through the warp. A
cluster was tried first and measured dropping 78% of the constraints, the nearest one
0.01m from those it kept -- cutting a continuum rather than rejecting aliasing.

**No consensus means the pose does not move.** Fewer than `FUSE_MIN` constraints
agreeing is not a reason to solve over the disagreeing ones anyway -- an empty list is
returned, and the solver hands back the transform it was given (checked: Ceres
converges on zero residuals and `optimized_parameters[0]` comes back untouched). Odom
carries the pose until a majority forms, which is what odom is for.
"""
import os

import numpy as np

#: Metres of odom travel after which a constraint is stale. Travel and not time:
#: drift comes with travel, so standing still expires nothing -- which is what keeps a
#: stationary robot from ending up with no constraints at all.
FUSE_MAX_M = float(os.environ.get("TINYNAV_FUSE_MAX_M", "3.0"))
#: How close two constraints' implied map->odom must sit to be the same answer. Wide
#: enough for PnP noise and for the drift across one window (measured on 122: an honest
#: correction is p50 0.20m, p90 0.32m), far under the corridor aliasing it separates
#: (5m and up on `home-to-n2-1`, whose 687 keyframes lie along a single 35m line).
#:
#: The transform this compares is NOT the same everywhere in the map: the map's own
#: keyframe poses are a VIO trajectory, so the map is locally consistent and globally
#: warped, and honest observations from stretches far apart imply genuinely different
#: transforms. Which is why the staleness cut comes FIRST and this only ever compares
#: constraints from the last few metres, where the warp is nothing. Reversing the two
#: would make honest disagreement look like aliasing.
FUSE_AGREE_M = float(os.environ.get("TINYNAV_FUSE_AGREE_M", "0.5"))
#: How many must agree before the solve is allowed to move the pose. Below this there
#: is no majority, only observations -- and one observation is one PnP with no
#: averaging left to damp it.
FUSE_MIN = int(os.environ.get("TINYNAV_FUSE_MIN", "5"))
#: Upstream's cap, kept as the upper bound.
FUSE_MAX = int(os.environ.get("TINYNAV_FUSE_MAX", "100"))


def _theil_sen(s, y):
    """Robust line through (s, y): the median of all pairwise slopes, then the median
    intercept. Least squares would be dragged by the very constraints this exists to
    exclude -- a block of aliased matches tilts a least-squares line until they fit it.
    Theil-Sen ignores an outlying minority instead of averaging it in.

    `s` is metres travelled, `y` one axis of the implied transform, so the slope is
    metres of transform per metre driven.
    """
    n = len(s)
    i, j = np.triu_indices(n, k=1)
    ds = s[j] - s[i]
    ok = np.abs(ds) > 1e-6
    if not ok.any():
        # Standing still: no baseline to measure a slope over, and none is needed.
        return 0.0, float(np.median(y))
    slope = float(np.median((y[j][ok] - y[i][ok]) / ds[ok]))
    return slope, float(np.median(y - slope * s))


def _agreeing(constraints, odom_poses):
    """The constraints that lie on the window's own regression line, newest last.

    **A line, not a ball around one of them.** The transform genuinely moves across a
    window: the map's keyframe poses are themselves a VIO trajectory, so the map is
    locally consistent and globally warped and the true transform slides as the robot
    travels through that warp. Measured on 122 on 2026-09-03, a ball dropped 78% of the
    constraints and the nearest one it dropped sat 0.01m from the ones it kept -- it was
    not rejecting aliasing, it was cutting a continuum in half.

    So what must agree is the TREND, and a constraint belongs if it sits within
    FUSE_AGREE_M of it. An aliased match is metres off the line however the line is
    tilted; a stretch of honest drift is on it end to end.
    """
    if len(constraints) < 2:
        return list(constraints)
    t = np.asarray([np.asarray(c[2])[:3, 3] for c in constraints])
    p = np.asarray([np.asarray(T)[:3, 3] for T in odom_poses])
    # Path length along the window, which is what the warp is a function of. Cumulative
    # rather than distance-from-the-end: a robot that turns back would give two
    # different constraints the same parameter and the line could not tell them apart.
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    fit = np.empty_like(t)
    for k in range(3):
        slope, intercept = _theil_sen(s, t[:, k])
        fit[:, k] = intercept + slope * s
    on_line = np.linalg.norm(t - fit, axis=1) <= FUSE_AGREE_M
    return [c for c, keep in zip(constraints, on_line) if keep]


def select_fusion_constraints(constraints, odom_poses):
    """The constraints taken within FUSE_MAX_M of driving of where we are now, reduced
    to those that agree about map->odom. Empty when no majority agrees.

    "Now" is the newest constraint's own odom, not `latest_odom_pose`: the solve this
    runs inside was triggered by that observation. `odom_poses[i]` is the odom pose
    constraint `i` was observed at -- the only thing the solver has no way to ask
    about. Returns the constraints to solve over, newest last.
    """
    if not constraints:
        return constraints
    here = np.asarray(odom_poses[-1])[:3, 3]
    travel = np.linalg.norm(
        np.asarray([np.asarray(T)[:3, 3] for T in odom_poses]) - here, axis=1)
    fresh = [c for c, d in zip(constraints, travel) if d <= FUSE_MAX_M]
    fresh_odom = [T for T, d in zip(odom_poses, travel) if d <= FUSE_MAX_M]
    agreed = _agreeing(fresh, fresh_odom)
    if len(agreed) < FUSE_MIN:
        return []
    return agreed[-FUSE_MAX:]
