"""Display actual planner/stair outputs in a robot-centred, heading-up view."""
import cv2
import numpy as np

CELLS, RES, SCALE, HEADER = 100, .05, 4, 34
BG = (20, 27, 37)


def render_bev(world):
    q = world.data.qpos[world.binding.root_qpos_adr:world.binding.root_qpos_adr + 7].copy()
    # MuJoCo free joint quaternion is wxyz.
    qw, qx, qy, qz = q[3:]
    yaw = np.arctan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    if hasattr(world,'sensors') and world.sensors.estimated is not None:
        T=world.sensors.estimated
        yaw=np.arctan2(T[1,2],T[0,2])
        q[0]=T[0,3]-.2*np.cos(yaw)
        q[1]=T[1,3]-.2*np.sin(yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    offset = (CELLS/2-.5-np.arange(CELLS))*RES
    forward, left = np.meshgrid(offset, offset, indexing='ij')
    xy = np.stack((q[0]+c*forward-s*left, q[1]+s*forward+c*left), axis=-1)

    def sample(grid, origin, resolution, fill):
        ij = np.floor((xy-np.asarray(origin)[:2])/resolution).astype(int)
        valid = ((ij >= 0) & (ij < np.asarray(grid.shape[:2]))).all(axis=-1)
        ix, iy = np.moveaxis(np.clip(ij, 0, np.asarray(grid.shape[:2])-1), -1, 0)
        out = grid[ix, iy].copy()
        out[~valid] = fill
        return out

    height = np.full((CELLS,CELLS,3), BG, np.uint8)
    planner = height.copy()
    result = world.bev_result
    known_count = 0
    if result:
        z = sample(result['height'],result['origin'],world.stair.gen.cfg.resolution,np.nan)
        known = np.isfinite(z)
        known_count = int(known.sum())
        ground = world.bev_cam_z-world.stair.gen.camera_height
        level = (np.clip((np.nan_to_num(z,nan=ground)-ground+1.5)/2.1,0,1)*255).astype(np.uint8)
        colored = cv2.applyColorMap(level,cv2.COLORMAP_TURBO)[...,::-1]
        height[known] = colored[known]
        walls = sample(result['obstacle'],result['origin'],world.stair.gen.cfg.resolution,False)
        height[walls] = (239,75,75)
    if world.bev_planner is not None:
        planner = sample(world.bev_planner,world.bev_planner_origin,world.planner.resolution,BG)
    if world.bev_mask is not None:
        mask = world.bev_mask
        obstacles = np.asarray(mask.data).reshape((mask.info.height,mask.info.width),order='F')>0
        blocked = sample(obstacles,(mask.info.origin.position.x,mask.info.origin.position.y),mask.info.resolution,False)
        planner[blocked] = (239,75,75)

    def pixel(point):
        dx,dy = np.asarray(point)[:2]-q[:2]
        return (int(round((CELLS/2-.5-(-s*dx+c*dy)/RES)*SCALE)),
                int(round((CELLS/2-.5-(c*dx+s*dy)/RES)*SCALE))+HEADER)

    def finish(image,title,legend):
        canvas = np.full((HEADER+CELLS*SCALE+68,CELLS*SCALE,3),BG,np.uint8)
        canvas[HEADER:HEADER+CELLS*SCALE]=cv2.resize(image,None,fx=SCALE,fy=SCALE,interpolation=cv2.INTER_NEAREST)
        for meter in range(-2,3):
            at=int((CELLS/2-.5-meter/RES)*SCALE)
            cv2.line(canvas,(0,HEADER+at),(CELLS*SCALE-1,HEADER+at),(75,83,95),1)
            cv2.line(canvas,(at,HEADER),(at,HEADER+CELLS*SCALE-1),(75,83,95),1)
        if len(world.bev_path)>1:
            # Avoid connecting separate candidate trajectories across large gaps.
            points=np.asarray(world.bev_path)
            splits=np.flatnonzero(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1)>.5)+1
            for segment in np.split(points,splits):
                if len(segment)>1:
                    cv2.polylines(canvas,[np.array([pixel(p) for p in segment],np.int32)],False,(70,235,255),2,cv2.LINE_AA)
        target=world.planner.target_pose
        if target is not None and np.isfinite(target[:2]).all():
            target_pixel=pixel(target)
            cv2.circle(canvas,target_pixel,6,(255,222,66),2,cv2.LINE_AA)
        center=pixel(q[:2])
        cv2.rectangle(canvas,(center[0]-12,center[1]-16),(center[0]+12,center[1]+16),(245,248,255),2)
        cv2.arrowedLine(canvas,center,(center[0],center[1]-28),(255,255,255),2,tipLength=.3)
        cv2.putText(canvas,title,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,242,255),1,cv2.LINE_AA)
        bottom=HEADER+CELLS*SCALE
        for i,line in enumerate(legend):
            cv2.putText(canvas,line,(8,bottom+18+i*19),cv2.FONT_HERSHEY_SIMPLEX,.38,(220,230,245),1,cv2.LINE_AA)
        return canvas

    height=finish(height,'Stair BEV | measured height',[
        'UP: forward | LEFT: robot left | grid: 1 m',
        'Dark: unobserved | red: stair wall cells',
        'Height: blue -1.5m -> red +0.6m from ground'])
    planner=finish(planner,'Planner BEV | clearance + obstacles',[
        'Actual /planning/height_map (ESDF 0-2 m)',
        'Red: blocked | cyan: selected path',
        'Yellow: target | white: robot | view: 5 x 5 m'])
    return height,planner,known_count
