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


class LocalRoute:
    """Cache local routing and measured obstacles; no recovery actions or stall timer."""
    replan_interval_s = .5
    goal_radius_m = .65
    max_linear_vel = .15

    def __init__(self):
        self.reset()

    def reset(self):
        self.target = None
        self.cells = {}
        self.next_replan = 0.0
        self.waypoint = None

    def observe(self, depth, transform, intrinsics, robot):
        stride = max(1,int(np.ceil(max(depth.shape[1]/160,depth.shape[0]/100))))
        image = depth[::stride*4,::stride*4]
        v,u = np.indices(image.shape)
        z = image.ravel()
        u,v = (u*stride*4).ravel(),(v*stride*4).ravel()
        valid = np.isfinite(z) & (z>0) & (z<=8)
        u,v,z = u[valid],v[valid],z[valid]
        points = np.column_stack(((u-intrinsics[0,2])/intrinsics[0,0]*z,
                                  (v-intrinsics[1,2])/intrinsics[1,1]*z,z))
        points = points @ transform[:3,:3].T + transform[:3,3]
        band = robot.obstacle
        points = points[(points[:,2]>=transform[2,3]+band.robot_z_bottom) &
                        (points[:,2]<=transform[2,3]+band.robot_z_top)]
        self.cells.update((tuple(cell),'blocked') for cell in np.floor(points[:,:2]/.1).astype(int))
        if len(self.cells)>50000:
            center = transform[:2,3]/.1
            self.cells = {cell:state for cell,state in self.cells.items() if math.dist(cell,center)<120}

    def update(self, xy, target, now, esdf, origin, resolution, half_width, dilation_cells):
        if target is None:
            self.reset()
            return None
        if self.target is None or np.linalg.norm(np.asarray(target)-self.target)>.25:
            self.target = np.asarray(target).copy()
            self.next_replan = 0.0
            self.waypoint = None
        distance = math.dist(xy[:2],target[:2])
        if distance<=self.goal_radius_m:
            self.waypoint = None
            return None
        if now>=self.next_replan:
            field = retained_esdf(esdf,origin,resolution,self.cells,.1,dilation_cells)
            direction = (np.asarray(target)[:2]-xy[:2])/distance
            samples = np.asarray(xy)[:2]+np.arange(0,min(distance,2.)+.001,resolution)[:,None]*direction
            indices = np.floor((samples-origin[:2])/resolution).astype(int)
            valid = np.all((indices>=0) & (indices<field.shape),axis=1)
            indices = indices[valid]
            obstructed = np.any(field[indices[:,0],indices[:,1]] < half_width+.025)
            self.waypoint = local_detour_target(field,origin,resolution,xy,target,half_width) if obstructed else None
            self.next_replan = now+self.replan_interval_s
        return self.waypoint
