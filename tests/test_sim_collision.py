from tool.simulator.planning_scene import SimObject, robot_hits_objects


def test_free_at_origin_in_l_corridor_entry():
    robot = {"front_len": 0.2, "rear_len": 0.2, "half_width": 0.15}
    wall = SimObject("wall", "box", (1.8, -0.85, 0.65), (5.6, 0.3, 1.3))
    assert not robot_hits_objects([0.0, 0.0], 0.0, robot, [wall])


def test_overlap_when_center_inside_box():
    robot = {"front_len": 0.2, "rear_len": 0.2, "half_width": 0.15}
    box = SimObject("block", "box", (0.0, 0.0, 0.65), (1.0, 1.0, 1.3))
    assert robot_hits_objects([0.0, 0.0], 0.0, robot, [box])


def test_side_graze_is_a_hit():
    robot = {"front_len": 0.2, "rear_len": 0.2, "half_width": 0.15}
    # wall along y, robot at x just overlapping the +x face
    wall = SimObject("wall", "box", (0.25, 0.0, 0.65), (0.2, 2.0, 1.3))
    assert robot_hits_objects([0.0, 0.0], 0.0, robot, [wall])


if __name__ == "__main__":
    test_free_at_origin_in_l_corridor_entry()
    test_overlap_when_center_inside_box()
    test_side_graze_is_a_hit()
    print("ok")
