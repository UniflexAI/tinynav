"""Lightweight route-target selection shared by map navigation and WebSim."""
import numpy as np

def select_target_position(paths, closest_idx, pos):
    max_speed = 0.5
    accumulated_distance = 0.0
    start_point = pos[:3]
    target_position = paths[-1]
    for i in range(closest_idx, len(paths) - 1):
        accumulated_distance += np.linalg.norm(paths[i][:2] - start_point[:2])
        if accumulated_distance > max_speed * 5:
            target_position = paths[i]
            break
        start_point = paths[i]
    return target_position
