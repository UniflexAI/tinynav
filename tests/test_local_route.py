import os
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tinynav.core import local_route as lr

RES = 0.05
SHAPE = (100, 100)
ORIGIN = np.array([-2.5, -2.5])
HALF_W = 0.15
# East along y=0, then north at x=1, leaving the grid.
ROUTE = np.array([[-2.0, 0.0], [1.0, 0.0], [1.0, 4.0]])
CORNER = ROUTE[1]


def _dense(route, step=0.05):
    out = [route[0]]
    for a, b in zip(route[:-1], route[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return np.array(out)


def _centres():
    xs = ORIGIN[0] + (np.arange(SHAPE[0]) + 0.5) * RES
    ys = ORIGIN[1] + (np.arange(SHAPE[1]) + 0.5) * RES
    return np.meshgrid(xs, ys, indexing='ij')


def _plan(seen, obstacle=None, route=ROUTE):
    obstacle = np.zeros(SHAPE, bool) if obstacle is None else obstacle
    esdf = distance_transform_edt(~obstacle) * RES if obstacle.any() else np.full(SHAPE, 10.0)
    cost = lr.cell_cost(esdf, seen, HALF_W)
    return lr.local_route(_dense(route), route[0], cost, ORIGIN, RES), cost


def _cells(path):
    return np.floor((_dense(path) - ORIGIN) / RES).astype(int)


def test_a_corner_seen_empty_is_cut():
    path, _ = _plan(np.ones(SHAPE, bool))
    assert np.linalg.norm(path - CORNER, axis=1).min() > 0.5


def test_but_one_never_seen_is_driven_round():
    x, y = _centres()
    seen = ~((x < CORNER[0] - 0.1) & (y > CORNER[1] + 0.1))
    path, _ = _plan(seen)
    c = _cells(path)
    inside = np.all((c >= 0) & (c < SHAPE), axis=1)
    assert seen[c[inside, 0], c[inside, 1]].all()
    assert np.linalg.norm(path - CORNER, axis=1).min() < 0.2


def test_an_obstacle_on_the_route_is_passed_clear_of_the_body():
    x, y = _centres()
    wall = (np.abs(x + 0.4) < 0.1) & (np.abs(y) < 0.6)
    path, cost = _plan(np.ones(SHAPE, bool), wall)
    c = _cells(path)
    inside = np.all((c >= 0) & (c < SHAPE), axis=1)
    assert np.isfinite(cost[c[inside, 0], c[inside, 1]]).all()


def test_the_route_leaves_the_grid_as_itself():
    path, _ = _plan(np.ones(SHAPE, bool))
    dense = _dense(ROUTE)
    _, k = lr.local_target(dense, ROUTE[0], ORIGIN, SHAPE, RES)
    np.testing.assert_array_equal(path[-(len(dense) - k - 1):], dense[k + 1:])


def test_a_goal_inside_the_grid_is_reached_exactly():
    route = np.array([[-2.0, 0.0], [1.0, 0.0], [1.0, 1.53]])
    path, _ = _plan(np.ones(SHAPE, bool), route=route)
    np.testing.assert_array_equal(path[-1], route[-1])


def test_a_target_the_body_cannot_reach_is_aimed_short_of_on_the_route():
    x, y = _centres()
    dense = _dense(ROUTE)
    _, k = lr.local_target(dense, ROUTE[0], ORIGIN, SHAPE, RES)
    post = np.hypot(x - dense[k, 0], y - dense[k, 1]) < 0.2
    path, cost = _plan(np.ones(SHAPE, bool), post)
    # The longest stretch of the route the path ends on starts where it aimed.
    aim = next(j for j in range(1, k + 1)
               if np.array_equal(path[-(len(dense) - j):], dense[j:]))
    assert aim < k
    c = _cells(path[:len(path) - (len(dense) - aim) + 1])
    assert np.isfinite(cost[c[:, 0], c[:, 1]]).all()
