import heapq
import math

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


def local_detour_target(esdf, origin, resolution, xy, target, half_width):
    """Suggest a local waypoint; the trajectory scorer still validates the footprint."""
    shape = esdf.shape
    start = tuple(np.floor((np.asarray(xy)[:2] - origin[:2]) / resolution).astype(int))
    if any(v < 0 or v >= shape[i] for i, v in enumerate(start)):
        return None
    free = esdf >= half_width + .025
    if esdf[start] < resolution*.5:
        return None
    free[start] = True
    goal_xy = np.asarray(target)[:2]
    goal = tuple(np.clip(np.floor((goal_xy - origin[:2]) / resolution).astype(int), 0, np.array(shape)-1))
    costs = {start: 0.0}
    parents = {}
    queue = [(0.0, start)]
    best = start
    best_distance = math.dist((np.array(start)+.5)*resolution+origin[:2], goal_xy)
    while queue:
        cost, cell = heapq.heappop(queue)
        if cost != costs[cell]:
            continue
        distance = math.dist((np.array(cell)+.5)*resolution+origin[:2], goal_xy)
        if distance < best_distance:
            best, best_distance = cell, distance
        if cell == goal:
            best = cell
            break
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            nxt = cell[0]+dx, cell[1]+dy
            if not (0 <= nxt[0] < shape[0] and 0 <= nxt[1] < shape[1]) or not free[nxt]:
                continue
            if dx and dy and (not free[cell[0]+dx,cell[1]] or not free[cell[0],cell[1]+dy]):
                continue
            step = resolution * math.hypot(dx,dy) * (1 + .04 / max(float(esdf[nxt]),.05))
            new_cost = cost + step
            if new_cost < costs.get(nxt, math.inf):
                costs[nxt], parents[nxt] = new_cost, cell
                heapq.heappush(queue, (new_cost, nxt))
    if best == start or best_distance >= math.dist(xy[:2], goal_xy)-.15:
        return None
    endpoint = (np.array(best)+.5)*resolution+origin[:2]
    if costs[best] > 1.8*math.dist(xy[:2],endpoint)+.3:
        return None
    path = [best]
    while path[-1] != start:
        path.append(parents[path[-1]])
    path.reverse()
    length = 0.0
    chosen = path[-1]
    for prev, cell in zip(path,path[1:]):
        length += math.dist(prev,cell)*resolution
        if length >= .4:
            chosen = cell
            break
    waypoint = np.asarray(target,dtype=float).copy()
    waypoint[:2] = (np.array(chosen)+.5)*resolution+origin[:2]
    return waypoint


def retained_esdf(esdf, origin, resolution, cells, cell_resolution, dilation_cells):
    retained = np.zeros(esdf.shape,dtype=bool)
    for cell,state in cells.items():
        if state != 'blocked':
            continue
        lower = np.floor((np.array(cell)*cell_resolution-origin[:2])/resolution).astype(int)
        upper = np.ceil(((np.array(cell)+1)*cell_resolution-origin[:2])/resolution).astype(int)
        lo = np.maximum(lower,0)
        hi = np.minimum(upper,esdf.shape)
        if np.all(hi > lo):
            retained[lo[0]:hi[0],lo[1]:hi[1]] = True
    if dilation_cells:
        retained = binary_dilation(retained,iterations=dilation_cells)
    return distance_transform_edt(~(retained | (esdf < resolution*.5))).astype(np.float32)*resolution
