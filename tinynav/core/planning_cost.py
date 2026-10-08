"""The trajectory cost, as a function of per-candidate features, plus a replay of it.

planning_node computes the features (`COLUMNS`) for every candidate, picks with
`select`, and publishes the same table on `/planning/cost_terms` -- so a recorded
frame can be re-scored offline and must come out the same. That is what makes a cost
change testable against the frames it is meant to fix, instead of one rig walk at a
time.

numpy only: the replay runs on a laptop against pilot's `cost_trace` CSVs.

    python3 -m tinynav.core.planning_cost cost-r3-*.csv            # agreement, who decided, shadow
    python3 -m tinynav.core.planning_cost cost-r3-*.csv --frame 120  # one frame's table
"""
import csv
import json
import sys

import numpy as np

#: One row per collision-free candidate. `i` indexes the full library (lattice, then
#: the reverse vocabulary). Angles in rad, yaw rate in rad/s (world frame, from the
#: trajectory's own poses), distances in m.
#:   clr        ESDF clearance score, 0 outside safety_radius (unweighted)
#:   occ        step index of the minimum clearance
#:   remaining  route arc length still ahead of the end cell
#:   path_dev   worst distance of the centre from the route
#:   route_err  end heading off the route's direction there
#:   goal_err   end heading off the bearing to the target
#:   goal_d     3D distance, end to target; goal_dxy the same in xy
COLUMNS = ('i', 'vx', 'omega', 'yaw', 'clr', 'occ', 'remaining', 'path_dev',
           'route_err', 'goal_err', 'goal_d', 'goal_dxy')

#: The terms, in the order they are summed: the order fixes the float sum, and a
#: replay has to reproduce exact ties.
TERMS = ('clearance', 'positional', 'smooth', 'heading', 'turn', 'gate')

#: Inside this of the goal the heading terms stop ranking anything: the bearing is
#: noise there and turning achieves nothing.
HEADING_GOAL_M = 0.3


#: Inside this of the goal the robot may stand still whatever else is open: that is
#: arriving. The same 0.3 m inside which the heading terms stop ranking anything.
STANDSTILL_GOAL_M = 0.3


def standstill_penalty(is_standstill, can_move, dist_to_goal):
    """Bans standing still while some other row is open, short of the goal.

    Every freeze measured so far was a standstill winning on cost with rows clear
    around it: on 2026-09-29, 18 forward and 15 turn-in-place rows collision-free,
    and the robot stood for minutes, because a turn's start cost outweighed what the
    heading term paid for it. A standstill is kept for when nothing else is open --
    then it is the honest answer -- and for the goal itself.
    """
    if is_standstill and can_move and dist_to_goal > STANDSTILL_GOAL_M:
        return 1e9
    return 0.0


def reverse_gate_penalty(vx, should_reverse):
    """Keeps the armed family and bans the other one, as upstream's gate does.

    The vx=0 rows are banned with the rest while reverse is armed. A fork-only
    exemption for them was tried on 2026-09-18 and taken back out with this return
    to upstream's shape; arming is rare enough at a 0.10 m entry (3.2% of frames
    measured on 122) that the rows are available almost whenever they are wanted.
    """
    return 0.0 if (vx < 0.0) == should_reverse else 1e9


def route_band_fade(end_remaining_m, terminal_band_m):
    """How much of the route is still ahead, as 0..1 over the last `terminal_band_m`.

    One encoding of the band, because two terms hand over across it: outside it the
    route says which way to point and how much progress is left, inside it the route
    has run out (remaining_map saturates at 0 and can rank nothing) and the goal --
    its position and its bearing -- takes over. Written twice, the two would drift and
    the robot would be pulled toward two different headings on arrival.
    """
    return min(1.0, end_remaining_m / max(terminal_band_m, 1e-6))


def route_heading_penalty(weight, heading_err_rad, end_remaining_m, terminal_band_m):
    """What a candidate pays for not pointing the way the route runs."""
    return weight * heading_err_rad * route_band_fade(end_remaining_m, terminal_band_m)


def turn_in_place_penalty(vx, yaw_rate, last_yaw_rate, w_turn, w_reversal,
                          w_start=0.0, start_err_rad=0.0):
    """What a rotating vx=0 row pays: a start cost that shrinks as the robot points
    further from the target, then per rad/s of rotation, and more if it turns against
    the last selected rotation. Moving rows and a standstill pay nothing here.

    The start cost is `w_start * (1 - start_err_rad / pi)`: all of it facing the
    target, none of it facing away. Facing the target a spin has little heading to
    gain, so a small, noisy gain no longer outranks standing still; facing away the
    spin is what the robot needs, and costs nothing extra.

    Yaw rates are world-frame (rad/s), from the trajectory poses, not the lattice's
    omega param -- the reverse vocabulary and the lattice do not share its sign.
    `w_turn + w_reversal` has to stay under the heading term's reach per rad/s
    (w_route_heading * rollout duration), or a standstill outranks the turn that
    would clear a large heading error.
    """
    if abs(vx) > 1e-3 or abs(yaw_rate) < 1e-3:
        return 0.0
    cost = w_start * (1.0 - min(abs(start_err_rad), np.pi) / np.pi) + w_turn * abs(yaw_rate)
    if yaw_rate * last_yaw_rate < 0.0:
        cost += w_reversal * abs(yaw_rate)
    return cost


def is_standstill(vx, yaw_rate):
    return abs(vx) <= 1e-3 and abs(yaw_rate) < 1e-3


def can_move(rows, should_reverse):
    """Some row is collision-free, not a standstill, and not banned by the reverse
    gate. `rows` holds only collision-free candidates."""
    return any(not is_standstill(r['vx'], r['yaw'])
               and reverse_gate_penalty(r['vx'], should_reverse) == 0.0
               for r in rows)


def candidate_terms(r, ctx):
    """The cost of one candidate, term by term (`TERMS`).

    `ctx` is the frame: has_route, should_reverse, can_move, to_goal (m), start_err
    (rad, robot heading off the target bearing), last_vx / last_omega (lattice params
    of the last pick), last_yaw (its world yaw rate), band (route_terminal_band) and
    w, the weights by name.
    """
    w = ctx['w']
    fade = route_band_fade(r['remaining'], ctx['band'])
    # The two heading references hand over across the terminal band: the route's
    # direction while there is route left, the goal's bearing once there is not. A
    # hard switch would leave the vx=0 rows unranked in the band.
    heading = 0.0
    if r['goal_d'] > HEADING_GOAL_M:
        to_goal = w['route_heading'] * r['goal_err']
        if ctx['has_route']:
            heading = (route_heading_penalty(w['route_heading'], r['route_err'],
                                             r['remaining'], ctx['band'])
                       + (1.0 - fade) * to_goal)
        else:
            heading = to_goal
    if ctx['has_route']:
        # The terminal term arms inside the band, where remaining_map has saturated
        # at 0 and can no longer rank anything.
        positional = (w['route_progress'] * r['remaining']
                      + w['path_follow'] * r['path_dev']
                      + w['goal_terminal'] * (1.0 - fade) * r['goal_dxy'])
    else:
        # Both route maps are flat: rank on the raw target.
        positional = 100 * r['goal_d']
    return {
        'clearance': r['clr'] * w['clearance'],
        'positional': positional,
        'smooth': 10 * (abs(ctx['last_vx'] - r['vx']) + abs(ctx['last_omega'] - r['omega'])),
        'heading': heading,
        'turn': turn_in_place_penalty(r['vx'], r['yaw'], ctx['last_yaw'],
                                      w['turn_in_place'], w['turn_reversal'],
                                      w['turn_start'], ctx['start_err']),
        'gate': (reverse_gate_penalty(r['vx'], ctx['should_reverse'])
                 + standstill_penalty(is_standstill(r['vx'], r['yaw']),
                                      ctx['can_move'], ctx['to_goal'])),
    }


def total(terms):
    s = 0.0
    for k in TERMS:
        s += terms[k]
    return s


def select(rows, ctx):
    """Position in `rows` of the cheapest candidate; the first one on a tie."""
    costs = [total(candidate_terms(r, ctx)) for r in rows]
    return min(range(len(rows)), key=costs.__getitem__)


# ------------------------------------------------------------ time to go --- #
# The cost under trial: seconds still to go, in shadow -- computed and published
# beside the weighted cost above, which still drives.

#: Rollout steps (of the 0.1 s lattice) the time cost samples, and what each sample
#: holds; cam_d is the camera's distance to the target, the one arrival measures.
STEP_SAMPLES = (5, 10, 15, 20, 25, 30)
STEP_COLUMNS = ('t', 'remaining', 'path_dev', 'route_err', 'goal_err', 'goal_dxy', 'cam_d')
#: map_node's _ARRIVE_M: arrival is its rule, so the time to go is the time until it
#: fires. A test holds the two equal.
ARRIVE_M = 0.5
#: Metres of route one unit of clearance score is worth: the weighted cost's
#: w_clearance / w_route_progress.
CLEARANCE_M = 2.0
#: The last pick keeps its slot unless something is this much faster.
HYSTERESIS_S = 0.15


def time_to_go(r, ctx):
    """Seconds to the goal through candidate `r`.

    If the rollout arrives -- the camera inside ARRIVE_M of the target where the route
    has run out (the target is only the goal there; before, it is a carrot) -- the
    time is the first sample that does. Otherwise it is the whole rollout plus the
    distance left at v_nom and the heading left at w_nom. Clearance adds the time its
    score is worth in route metres either way.

    `ctx` adds v_nom (m/s) and w_nom (rad/s) to what candidate_terms reads.
    """
    v, w, band = ctx['v_nom'], ctx['w_nom'], ctx['band']
    clearance = CLEARANCE_M * r['clr'] / v
    for t, rem, dev, rerr, gerr, gd, cam_d in r['steps']:
        if cam_d < ARRIVE_M and (not ctx['has_route'] or rem <= band):
            return t + clearance
    t, rem, dev, rerr, gerr, gd, cam_d = r['steps'][-1]
    if ctx['has_route']:
        fade = route_band_fade(rem, band)
        d = rem + dev + (1.0 - fade) * gd
        h = fade * rerr + (1.0 - fade) * gerr
    else:
        d, h = gd, gerr
    return t + d / v + h / w + clearance


def _time_allowed(r, ctx):
    return (reverse_gate_penalty(r['vx'], ctx['should_reverse']) == 0.0
            and standstill_penalty(is_standstill(r['vx'], r['yaw']),
                                   ctx['can_move'], ctx['to_goal']) == 0.0)


def select_time(rows, ctx):
    """Position in `rows` of the time cost's pick. The gates filter, falling back to
    every row when they leave none; then hysteresis on (vx, omega) of the last pick,
    `ctx['shadow_last']`."""
    pool = [k for k, r in enumerate(rows) if _time_allowed(r, ctx)] or list(range(len(rows)))
    cost = {k: time_to_go(rows[k], ctx) for k in pool}
    best = min(pool, key=cost.__getitem__)
    last = ctx.get('shadow_last')
    if last is not None:
        for k in pool:
            if (abs(rows[k]['vx'] - last[0]) < 1e-3 and abs(rows[k]['omega'] - last[1]) < 1e-3
                    and cost[k] <= cost[best] + HYSTERESIS_S):
                return k
    return best


# ---------------------------------------------------------------- replay --- #
# RECORDING: everything below goes with /planning/cost_terms.

def read_frames(path):
    """The frames of one cost_trace CSV: `t,payload`, payload the published JSON.
    Skips the header and the round-seal line."""
    with open(path, newline='') as fh:
        for cells in csv.reader(fh):
            if len(cells) != 2 or cells[0] == 't':
                continue
            frame = json.loads(cells[1])
            frame['t'] = float(cells[0])
            frame['rows'] = [dict(zip(frame['cols'], v)) for v in frame['rows']]
            yield frame


def kind(r):
    if r['vx'] < 0.0:
        return 'reverse'
    if abs(r['vx']) > 1e-3:
        return 'forward'
    return 'standstill' if is_standstill(r['vx'], r['yaw']) else 'turn'


def _table(frame, top=8):
    rows, ctx = frame['rows'], frame
    scored = sorted(((total(t), r, t) for r in rows for t in [candidate_terms(r, ctx)]),
                    key=lambda x: x[0])
    print(f"t={frame['t']:.2f} sel={frame['sel']} to_goal={ctx['to_goal']:.2f} "
          f"start_err={np.rad2deg(ctx['start_err']):.0f}deg has_route={ctx['has_route']} "
          f"should_reverse={ctx['should_reverse']} can_move={ctx['can_move']}")
    print(f"{'i':>4} {'kind':>10} {'vx':>5} {'yaw':>6} {'total':>9} "
          + ' '.join(f'{k:>10}' for k in TERMS))
    for c, r, t in scored[:top]:
        mark = '*' if r['i'] == frame['sel'] else ' '
        print(f"{r['i']:>3}{mark} {kind(r):>10} {r['vx']:5.2f} {r['yaw']:6.2f} {c:9.1f} "
              + ' '.join(f'{t[k]:10.1f}' for k in TERMS))


def replay(paths, frame_no=None):
    """Re-scores every frame with the cost as it is now, and says how often it picks
    what the robot picked, what was picked, and -- whenever a turn or a standstill won
    with a forward row open -- which term decided against the best forward row."""
    n = agree = 0
    picked, decided = {}, {}
    for path in paths:
        for frame in read_frames(path):
            if frame_no is not None:
                if n == frame_no:
                    _table(frame)
                    return
                n += 1
                continue
            rows = frame['rows']
            ctx = frame
            n += 1
            mine = rows[select(rows, ctx)]
            if mine['i'] == frame['sel']:
                agree += 1
            else:
                logged = next(r for r in rows if r['i'] == frame['sel'])
                gap = total(candidate_terms(logged, ctx)) - total(candidate_terms(mine, ctx))
                print(f"{path}: frame {n - 1} t={frame['t']:.2f} logged {frame['sel']} "
                      f"replayed {mine['i']} gap {gap:.4g}")
            k = kind(mine)
            picked[k] = picked.get(k, 0) + 1
            if k in ('turn', 'standstill'):
                fwd = [r for r in rows if kind(r) == 'forward'
                       and reverse_gate_penalty(r['vx'], ctx['should_reverse']) == 0.0]
                if fwd:
                    best = min(fwd, key=lambda r: total(candidate_terms(r, ctx)))
                    a, b = candidate_terms(mine, ctx), candidate_terms(best, ctx)
                    term = max(TERMS, key=lambda t: b[t] - a[t])
                    decided[term] = decided.get(term, 0) + 1
    if frame_no is not None:
        print(f'only {n} frames')
        return
    print(f'{n} frames, {agree} replayed to the logged pick ({100.0 * agree / max(n, 1):.1f}%)')
    shadow(paths)
    print('picked: ' + ' '.join(f'{k}={v}' for k, v in sorted(picked.items())))
    if decided:
        print('turn/standstill over an open forward row, the term that cost forward most: '
              + ' '.join(f'{k}={v}' for k, v in sorted(decided.items(), key=lambda x: -x[1])))


def shadow(paths):
    """The time cost against the weighted one on the same frames: how often its
    replay reproduces the shadow pick the planner logged, and where the two costs
    pick a different kind of row."""
    n = agree = 0
    flips = {}
    for path in paths:
        for frame in read_frames(path):
            if 'shadow' not in frame:
                continue
            rows = frame['rows']
            n += 1
            mine = rows[select_time(rows, frame)]
            agree += mine['i'] == frame['shadow']
            a, b = kind(next(r for r in rows if r['i'] == frame['sel'])), kind(mine)
            if a != b:
                flips[(a, b)] = flips.get((a, b), []) + [n - 1]
    if not n:
        return
    print(f'shadow: {n} frames, {agree} replayed to the logged shadow pick '
          f'({100.0 * agree / n:.1f}%)')
    for (a, b), frames in sorted(flips.items(), key=lambda x: -len(x[1])):
        print(f'  weighted {a} -> time {b}: {len(frames)} frames, e.g. {frames[:12]}')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    frame_no = None
    if '--frame' in argv:
        k = argv.index('--frame')
        frame_no = int(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    if not argv:
        print(__doc__)
        return 2
    replay(argv, frame_no)
    return 0


if __name__ == '__main__':
    sys.exit(main())
