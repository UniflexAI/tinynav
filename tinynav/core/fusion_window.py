"""How many relocalization observations the solve averages over.

`pose_graph_solve` is not a graph on this path: node 1 is held constant at identity
and every constraint is `(0, 1, observation_T)`, so there is one free variable and N
direct measurements of it. The solve is a weighted mean on SE(3), which makes this
module the whole of the fusion -- there is no structure here for a wider window to
exploit.

Each constraint is one observation's implied map->odom. Upstream solves over the
newest 100, which at ~0.8 relocalizations a second reaches back two minutes. That
is an average, and an average is only right if the observations are a constant plus
noise. They are not: the map's own keyframe poses are a VIO trajectory, so the map
is locally consistent and globally warped, and the true transform slides as the
robot travels through the warp. Against a ramp an average **lags reality by half
the window** -- measured on 118 standing still, 20 consecutive PnP answers agreed
on (42.44, -4.47) while the fused estimate sat 0.75m away, because a good
observation was one vote against ninety-nine older ones.

So the window is 1: the newest observation IS the transform, and nothing is averaged.

**Averaging cannot help here anyway**, because the error is not noise. Measured on
122 (2026-09-03) on `home-to-n2-1`, one straight 35m corridor: PnP returns two nearly
equally good answers 4-9m apart -- 72 inliers against 72 for a place 4.6m away -- and
which one wins flips. Averaging a bimodal distribution gives the point between the two
modes, which is nowhere; a LARGE window also drags the estimate behind the robot,
since the old mode keeps most of the votes. At window 3 the pose moved p50 1.21m over
19 consecutive solves, following the flip. Neither size is right, and no size is.

What separates the two modes is odometry, not the observations: `pilot/nav/jump.py`
refuses a transform that moved further than the robot rode. **That is the only
discriminator in the path**, which is why this module no longer tries to be one.

**Nothing here judges an observation**, and three attempts to were measured failing on
2026-09-03. Refusing what disagreed with the current estimate defended a wrong pose
through 157 consecutive refusals over 20m, 104 of them the truth. Letting the rejected
ones overrule it when enough agreed believed ten aliased matches that agreed with each
other to centimetres and moved the pose 49.65m. Requiring a new observation to lie on
the window's own robust regression line failed for the same reason: **a run of aliased
matches is itself a smooth line** -- the robot drives straight down the corridor and so
does the wrong hypothesis -- so the test rejects lone outliers and admits exactly the
correlated runs that do the damage.
"""
import os

#: How many of the newest observations the solve averages over. 1 means the newest
#: observation IS the transform, and averaging is off.
#:
#: `pose_graph_solve` is not a graph here: node 1 is held constant at identity and
#: every constraint is `(0, 1, observation_T)`, so there is one free variable and N
#: direct measurements of it. The solve is a weighted mean on SE(3) and nothing else,
#: which makes this parameter the whole of it -- and the average is what lags, because
#: the observations are not a constant plus noise. See the module docstring.
FUSE_WINDOW = int(os.environ.get("TINYNAV_FUSE_WINDOW", "1"))


def select_fusion_constraints(constraints, odom_poses):
    """The newest `FUSE_WINDOW` constraints, newest last.

    `odom_poses[i]` is the odom pose constraint `i` was observed at. Unused here --
    it is the argument the travel-bounded window needed, kept because the caller in
    `map_node.py` has it and a window that wants it again should not have to thread
    it back through.
    """
    if not constraints or FUSE_WINDOW <= 0:
        return constraints
    return constraints[-FUSE_WINDOW:]
