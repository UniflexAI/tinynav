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

**There are two error scales here and only one of them is this module's.** Measured on
122 (2026-09-03) on `home-to-n2-1`, standing still for eleven minutes -- so the true
transform was constant and every difference between observations was measurement error:

    ~0.3m   consecutive observations disagree by p50 0.3m, commonly 0.65m. Real
            noise. Averaging removes it, and this is what the window is for.
    ~4.3m   occasionally an observation lands on a specific other keyframe 4.3m
            along the corridor -- 72 inliers against 72 for that place. Not noise,
            and averaging does the worst possible thing with it: over N it dilutes
            one flip to 4.3/N metres, small enough to pass a veto and still wrong.

So the window is sized for the noise alone: 5 observations, which brings the p50 0.3m
scatter to roughly 0.15m. It is NOT sized to fight the flip, and cannot be. At 100 the
average lags the robot; at 3 it followed the flip, p50 1.21m of pose movement over 19
consecutive solves; at 1 the raw 0.3m noise reached the pose and `pilot/nav/jump.py`
refused 42% of solves, most of them honest.

**The flip is answered before it becomes an observation**, by the candidate radius in
`pilot/nav/candidates.py`: a 3.0m radius does not retrieve a keyframe 4.3m away, so the
alternative never enters. `pilot/nav/jump.py` then refuses only what neither layer
catches. Three layers, three scales -- and this one is the noise.

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

#: How many of the newest observations the solve averages over. Sized off the
#: measured single-observation noise (p50 0.3m standing still on 122), not off the
#: aliasing -- see the module docstring for why no size answers the aliasing.
#:
#: `pose_graph_solve` is not a graph here: node 1 is held constant at identity and
#: every constraint is `(0, 1, observation_T)`, so there is one free variable and N
#: direct measurements of it. The solve is a weighted mean on SE(3) and nothing else,
#: which makes this parameter the whole of it -- and the average is what lags, because
#: the observations are not a constant plus noise. See the module docstring.
FUSE_WINDOW = int(os.environ.get("TINYNAV_FUSE_WINDOW", "5"))


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
