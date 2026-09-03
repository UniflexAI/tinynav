"""How many relocalization observations the pose-graph solve averages over.

Each constraint is one observation's implied map->odom. Upstream solves over the
newest 100, which at ~0.8 relocalizations a second reaches back two minutes. That
is an average, and an average is only right if the observations are a constant plus
noise. They are not: the map's own keyframe poses are a VIO trajectory, so the map
is locally consistent and globally warped, and the true transform slides as the
robot travels through the warp. Against a ramp an average **lags reality by half
the window** -- measured on 118 standing still, 20 consecutive PnP answers agreed
on (42.44, -4.47) while the fused estimate sat 0.75m away, because a good
observation was one vote against ninety-nine older ones.

So the window is small: the newest `FUSE_WINDOW` observations, and no more. Small
enough that the lag is nothing, still more than one so a single PnP is damped.

**Nothing here judges an observation.** Two attempts to were measured failing in
opposite directions on 2026-09-03: refusing what disagreed with the current estimate
defended a wrong pose through 157 consecutive refusals over 20m, 104 of them the
truth; letting the rejected ones overrule it when enough agreed believed ten aliased
matches that agreed with each other to centimetres and moved the pose 49.65m. A
trend-consensus filter over a 3m travel window was tried in their place and is what
this replaced -- it starved the solve exactly where the fix rate collapses, and the
corridor aliasing it was aimed at is answered upstream of here, by the candidate
radius in `pilot/nav/candidates.py`. The odometry veto in `pilot/nav/jump.py` is the
only thing that refuses a result, and it refuses the fused pose, not an observation.
"""
import os

#: How many of the newest observations the solve averages over. A count and not a
#: distance: what has to stay small is the number of votes an old observation gets,
#: and the lag that number buys is half the window whether the robot is moving or not.
FUSE_WINDOW = int(os.environ.get("TINYNAV_FUSE_WINDOW", "3"))


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
