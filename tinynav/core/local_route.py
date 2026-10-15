"""The line the trajectory scoring follows inside the planner's grid, decided by what the
camera has seen rather than by the global route alone.

The global route says where to go. Inside the grid the scoring follows the cheapest path
to the last route point the grid still holds, through cells weighted by what is known of
them: a cell a ray has passed through costs 1, a cell nothing has looked at costs
`unknown_cost`, and a cell within `clear_m` of an obstacle costs more the closer it is.
So the inside of a corner is cut once it has been seen to be empty, and not before.

Pure functions over the planner's arrays; index [ix, iy] with cell centres at
origin + (i + 0.5) * resolution, as in `build_route_fields`.
"""
import heapq

import numpy as np
from numba import njit

#: How many times a cell no ray has passed through costs over one seen to be free.
UNKNOWN_COST = 3.0
#: Clearance from the body's side below which a cell costs more, up to 1 + CLEAR_WEIGHT
#: at contact. Zero beyond it, so an open corridor has no pull towards its middle.
CLEAR_M = 0.3
CLEAR_WEIGHT = 1.0
#: The local target is the last route point at least this far inside the grid's edge.
EDGE_M = 0.3
#: A voxel a ray has passed through reads below this (each pass takes 0.05 off, and the
#: grid decays by 0.99 a frame).
FREE_BELOW = -0.01


def observed_free(occupancy_grid):
    """(h, w) bool: a ray has passed through some voxel of the column. The grid spans
    exactly the body band, so every layer counts."""
    return np.any(occupancy_grid < FREE_BELOW, axis=2)


def cell_cost(esdf, seen, half_width, unknown_cost=UNKNOWN_COST, clear_m=CLEAR_M,
              clear_weight=CLEAR_WEIGHT):
    """Per-cell traversal cost, inf where the body's side would touch an obstacle.
    `esdf` is the distance (m) from each cell to the nearest obstacle cell."""
    clr = esdf - half_width
    near = np.clip((clear_m - clr) / clear_m, 0.0, 1.0) if clear_m > 0 else 0.0
    cost = np.where(seen, 1.0, unknown_cost) * (1.0 + clear_weight * near)
    return np.where(clr < 0.0, np.inf, cost)


def local_target(route_xy, here_xy, origin_xy, shape, resolution, edge_m=EDGE_M):
    """(nearest, last): the index of the route point nearest `here_xy`, and of the last
    one from there on before the route first leaves the grid's interior. None when the
    nearest one is outside."""
    route_xy = np.asarray(route_xy, dtype=float)
    lo = np.asarray(origin_xy, dtype=float) + edge_m
    hi = np.asarray(origin_xy, dtype=float) + np.asarray(shape, dtype=float) * resolution - edge_m
    inside = np.all((route_xy >= lo) & (route_xy < hi), axis=1)
    k0 = int(np.argmin(np.linalg.norm(route_xy - here_xy, axis=1)))
    if not inside[k0]:
        return None
    out = np.flatnonzero(~inside[k0:])
    return k0, (k0 + int(out[0]) - 1 if len(out) else len(route_xy) - 1)


@njit(cache=True)
def _dijkstra(cost, s0, s1):
    n0, n1 = cost.shape
    dist = np.full((n0, n1), np.inf)
    parent = np.full((n0, n1), -1, dtype=np.int64)
    dist[s0, s1] = 0.0
    heap = [(0.0, np.int64(s0 * n1 + s1))]
    while len(heap) > 0:
        d, k = heapq.heappop(heap)
        x, y = k // n1, k % n1
        if d > dist[x, y]:
            continue
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= n0 or ny >= n1 or not np.isfinite(cost[nx, ny]):
                    continue
                nd = d + np.sqrt(dx * dx + dy * dy) * cost[nx, ny]
                if nd < dist[nx, ny]:
                    dist[nx, ny] = nd
                    parent[nx, ny] = k
                    heapq.heappush(heap, (nd, np.int64(nx * n1 + ny)))
    return dist, parent


@njit(cache=True)
def _line_max(cost, a0, a1, b0, b1):
    n = int(max(abs(b0 - a0), abs(b1 - a1))) * 2 + 1
    worst = 0.0
    for i in range(n + 1):
        t = i / n
        c = cost[int(round(a0 + t * (b0 - a0))), int(round(a1 + t * (b1 - a1)))]
        if c > worst:
            worst = c
    return worst


@njit(cache=True)
def _pull(cost, cells):
    """Drop corners of the cell chain where a straight line crosses nothing dearer than
    the stretch of chain it replaces, so the route's heading is not the 8-connected
    staircase's."""
    keep = [0]
    i, n = 0, len(cells)
    while i < n - 1:
        j, worst = i + 1, max(cost[cells[i, 0], cells[i, 1]], cost[cells[i + 1, 0], cells[i + 1, 1]])
        while j + 1 < n:
            w = max(worst, cost[cells[j + 1, 0], cells[j + 1, 1]])
            if _line_max(cost, cells[i, 0], cells[i, 1], cells[j + 1, 0], cells[j + 1, 1]) > w + 1e-9:
                break
            j, worst = j + 1, w
        keep.append(j)
        i = j
    return np.array(keep)


def local_route(route_xy, here_xy, cost, origin_xy, resolution, edge_m=EDGE_M):
    """The route the scoring should follow: the cheapest path over `cost` from
    `here_xy` to the local target, then the global route on from there. None when the
    route has no point inside the grid near the robot or nothing finite is reachable;
    the caller then keeps the global route."""
    route_xy = np.asarray(route_xy, dtype=float)
    origin_xy = np.asarray(origin_xy, dtype=float)
    span = local_target(route_xy, here_xy, origin_xy, cost.shape, resolution, edge_m)
    if span is None:
        return None
    s = np.floor((np.asarray(here_xy, dtype=float) - origin_xy) / resolution).astype(np.int64)
    if np.any(s < 0) or np.any(s >= cost.shape):
        return None
    dist, parent = _dijkstra(np.ascontiguousarray(cost, dtype=np.float64), s[0], s[1])
    # A target in or behind something: aim at the last route point before it that can
    # be reached, never off the route.
    cells = np.floor((route_xy[span[0]:span[1] + 1] - origin_xy) / resolution).astype(np.int64)
    ok = np.flatnonzero(np.isfinite(dist[cells[:, 0], cells[:, 1]]))
    if len(ok) == 0 or ok[-1] == 0:
        return None
    k, g = span[0] + int(ok[-1]), cells[ok[-1]]
    n1 = cost.shape[1]
    chain = [g[0] * n1 + g[1]]
    while chain[-1] != s[0] * n1 + s[1]:
        chain.append(parent[chain[-1] // n1, chain[-1] % n1])
    cells = np.array([(c // n1, c % n1) for c in chain[::-1]], dtype=np.int64)
    # Leaving the start cell is never what makes a line dear, even when the robot
    # stands within reach of something.
    pull_cost = np.where(np.isfinite(cost), cost, 1e9)
    pull_cost[s[0], s[1]] = 1.0
    cells = cells[_pull(pull_cost, cells)]
    path = origin_xy + (cells + 0.5) * resolution
    path[0] = here_xy
    path[-1] = route_xy[k]  # the route point itself, not its cell centre
    return np.vstack([path, route_xy[k + 1:]])
