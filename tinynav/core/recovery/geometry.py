"""Conservative swept-footprint geometry shared by recovery adapters."""
import math


def footprint_cells(xy, robot, resolution):
    radius = math.hypot(robot.get('length',.4)/2+abs(robot.get('control_x',0)),
                        robot.get('width',.3)/2+abs(robot.get('control_y',0)))
    if robot.get('shape')=='circle':radius=max(radius,robot['radius'])
    radius += robot.get('recovery_margin',0)
    n = math.ceil(radius/resolution)
    return {(math.floor((xy[0]+i*resolution)/resolution),math.floor((xy[1]+j*resolution)/resolution))
            for i in range(-n,n+1) for j in range(-n,n+1) if math.hypot(i*resolution,j*resolution)<=radius+resolution/2}
