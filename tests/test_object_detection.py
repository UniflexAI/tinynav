import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tinynav', 'core'))
from planning_node import (
    apply_object_hits,
    decay_object_grids,
    label_occupied_column,
    lookup_transform,
    project_color_detections_to_voxels,
    roll_object_grids,
    store_transform,
)
from tinynav.core.models_trt import COCO_CLASS_NAMES, coco_class_ids, decode_yolo_output


def test_roll_object_grids_shifts_and_marks_exposed_slab_unknown():
    grid_shape = (4, 4, 2)
    class_grid = np.full(grid_shape, -1, dtype=np.int16)
    ttl_grid = np.zeros(grid_shape, dtype=np.int16)
    class_grid[1, 1, 0] = 7
    ttl_grid[1, 1, 0] = 5

    old_origin = np.array([0.0, 0.0, 0.0])
    new_origin = np.array([1.0, 0.0, 0.0])
    rolled_class, rolled_ttl, updated_origin = roll_object_grids(
        class_grid, ttl_grid, old_origin, new_origin, resolution=1.0,
    )

    np.testing.assert_allclose(updated_origin, new_origin)
    assert rolled_class[0, 1, 0] == 7
    assert rolled_ttl[0, 1, 0] == 5
    assert np.all(rolled_class[-1, :, :] == -1)
    assert np.all(rolled_ttl[-1, :, :] == 0)


def test_roll_object_grids_is_noop_without_movement():
    grid_shape = (3, 3, 3)
    class_grid = np.full(grid_shape, -1, dtype=np.int16)
    class_grid[0, 0, 0] = 3
    ttl_grid = np.zeros(grid_shape, dtype=np.int16)
    ttl_grid[0, 0, 0] = 9
    origin = np.array([0.0, 0.0, 0.0])

    rolled_class, rolled_ttl, updated_origin = roll_object_grids(
        class_grid, ttl_grid, origin, origin.copy(), resolution=0.5,
    )

    np.testing.assert_array_equal(rolled_class, class_grid)
    np.testing.assert_array_equal(rolled_ttl, ttl_grid)
    np.testing.assert_allclose(updated_origin, origin)


def test_label_occupied_column_tags_existing_occupied_cells():
    occupancy_grid = np.zeros((4, 4, 5))
    occupancy_grid[2, 3, 1] = 0.5
    occupancy_grid[2, 3, 3] = 0.5

    hits = label_occupied_column(
        occupancy_grid, class_id=7, x=2.5, y=3.5, origin=(0.0, 0.0, 0.0), resolution=1.0, occ_threshold=0.1,
    )

    assert hits.shape == (2, 4)
    np.testing.assert_array_equal(hits[:, :3], [[2, 3, 1], [2, 3, 3]])
    assert np.all(hits[:, 3] == 7)


def test_label_occupied_column_empty_when_out_of_bounds_or_unoccupied():
    occupancy_grid = np.zeros((4, 4, 5))
    occupancy_grid[2, 3, 1] = 0.5

    out_of_bounds = label_occupied_column(
        occupancy_grid, class_id=7, x=100.0, y=100.0, origin=(0.0, 0.0, 0.0), resolution=1.0, occ_threshold=0.1,
    )
    assert out_of_bounds.shape == (0, 4)

    empty_column = label_occupied_column(
        occupancy_grid, class_id=7, x=0.5, y=0.5, origin=(0.0, 0.0, 0.0), resolution=1.0, occ_threshold=0.1,
    )
    assert empty_column.shape == (0, 4)


def test_store_and_lookup_transform_direct_edge():
    edges = {}
    T_ab = np.eye(4)
    T_ab[:3, 3] = [1.0, 2.0, 3.0]
    store_transform(edges, 'A', 'B', T_ab)

    np.testing.assert_allclose(lookup_transform(edges, 'A', 'B'), T_ab, atol=1e-5)
    np.testing.assert_allclose(lookup_transform(edges, 'B', 'A'), np.linalg.inv(T_ab), atol=1e-5)


def test_lookup_transform_chains_multiple_hops():
    edges = {}
    T_ab = np.eye(4); T_ab[:3, 3] = [1.0, 0.0, 0.0]
    T_bc = np.eye(4); T_bc[:3, 3] = [0.0, 2.0, 0.0]
    store_transform(edges, 'A', 'B', T_ab)
    store_transform(edges, 'B', 'C', T_bc)

    T_ac = lookup_transform(edges, 'A', 'C')
    np.testing.assert_allclose(T_ac, T_ab @ T_bc, atol=1e-5)


def test_lookup_transform_same_frame_is_identity():
    np.testing.assert_allclose(lookup_transform({}, 'A', 'A'), np.eye(4))


def test_lookup_transform_unresolved_returns_none():
    edges = {}
    store_transform(edges, 'A', 'B', np.eye(4))
    assert lookup_transform(edges, 'A', 'Z') is None
    assert lookup_transform({}, 'A', 'B') is None


def test_project_color_detections_to_voxels_identity_extrinsic():
    # Same geometry as label_occupied_column's own tests: a 2x2 pixel box
    # around the principal point backprojects to a horizontal anchor landing
    # in voxel column (3, 3). With T_depth_color = identity and color_K ==
    # depth_K, color-space pixel coords equal depth-space pixel coords, so
    # this should reproduce the same column as the (removed) grayscale path.
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    depth_K = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]])
    color_K = depth_K
    T_cam_to_world = np.eye(4)
    T_depth_color = np.eye(4)
    color_shape = (20, 20, 3)
    origin = np.array([-2.0, -2.0, -2.0])
    resolution = 0.5

    occupancy_grid = np.zeros((8, 8, 10))
    occupancy_grid[3, 3, 5] = 0.5
    occupancy_grid[3, 3, 8] = 0.5
    occupancy_grid[0, 0, 0] = 0.5  # decoy, unrelated column

    detections = [(7, 0.9, 9.0, 9.0, 11.0, 11.0)]
    hits = project_color_detections_to_voxels(
        detections, depth, T_cam_to_world, depth_K, color_K, T_depth_color, color_shape,
        occupancy_grid, origin, resolution, step=1, occ_threshold=0.1,
    )

    assert hits.shape == (2, 4)
    np.testing.assert_array_equal(sorted(hits[:, :3].tolist()), [[3, 3, 5], [3, 3, 8]])
    assert np.all(hits[:, 3] == 7)


def test_project_color_detections_to_voxels_applies_nonidentity_extrinsic():
    # Single valid depth pixel at (u=5, v=5), depth=4 -> depth-frame point
    # (-2, -2, 4). A +1m X translation from depth to color frame moves it to
    # (-1, -2, 4), which color_K projects to (color_u, color_v) = (7.5, 5.0).
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[5, 5] = 4.0
    depth_K = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]])
    color_K = depth_K
    T_cam_to_world = np.eye(4)
    T_depth_color = np.eye(4)
    T_depth_color[0, 3] = 1.0
    color_shape = (20, 20, 3)
    origin = np.array([-5.0, -5.0, -5.0])
    resolution = 1.0

    occupancy_grid = np.zeros((10, 10, 10))
    occupancy_grid[3, 3, 9] = 0.5
    occupancy_grid[0, 0, 0] = 0.5  # decoy, unrelated column

    # Box tightly containing the expected (7.5, 5.0) projection.
    matching = [(3, 0.9, 7.0, 4.0, 8.0, 6.0)]
    hits = project_color_detections_to_voxels(
        matching, depth, T_cam_to_world, depth_K, color_K, T_depth_color, color_shape,
        occupancy_grid, origin, resolution, step=1, occ_threshold=0.1,
    )
    assert hits.shape == (1, 4)
    np.testing.assert_array_equal(hits[0], [3, 3, 9, 3])

    # A box that does NOT contain the (shifted) projection gets no hits, even
    # though it would have matched under an identity extrinsic (proving the
    # transform is actually being applied, not skipped).
    non_matching = [(3, 0.9, 0.0, 0.0, 1.0, 1.0)]
    hits2 = project_color_detections_to_voxels(
        non_matching, depth, T_cam_to_world, depth_K, color_K, T_depth_color, color_shape,
        occupancy_grid, origin, resolution, step=1, occ_threshold=0.1,
    )
    assert hits2.shape == (0, 4)


def test_project_color_detections_to_voxels_handles_empty_and_out_of_frame():
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    depth_K = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]])
    color_K = depth_K
    T_cam_to_world = np.eye(4)
    T_depth_color = np.eye(4)
    color_shape = (20, 20, 3)
    occupancy_grid = np.full((8, 8, 10), 0.5)  # everywhere occupied, to isolate the framing checks
    origin = np.array([-2.0, -2.0, -2.0])
    resolution = 0.5

    empty = project_color_detections_to_voxels(
        [], depth, T_cam_to_world, depth_K, color_K, T_depth_color, color_shape, occupancy_grid, origin, resolution,
    )
    assert empty.shape == (0, 4)

    # Box entirely outside the color image bounds is skipped, not an error.
    out_of_frame = [(1, 0.5, 100.0, 100.0, 110.0, 110.0)]
    hits = project_color_detections_to_voxels(
        out_of_frame, depth, T_cam_to_world, depth_K, color_K, T_depth_color, color_shape, occupancy_grid, origin, resolution,
    )
    assert hits.shape == (0, 4)


def test_apply_object_hits_writes_class_and_refreshes_ttl():
    class_grid = np.full((3, 3, 3), -1, dtype=np.int16)
    ttl_grid = np.zeros((3, 3, 3), dtype=np.int16)
    hits = np.array([[1, 1, 1, 42], [2, 2, 2, 42]], dtype=np.int32)

    apply_object_hits(class_grid, ttl_grid, hits, ttl_frames=10)

    assert class_grid[1, 1, 1] == 42 and ttl_grid[1, 1, 1] == 10
    assert class_grid[2, 2, 2] == 42 and ttl_grid[2, 2, 2] == 10
    assert class_grid[0, 0, 0] == -1 and ttl_grid[0, 0, 0] == 0


def test_apply_object_hits_noop_on_empty_hits():
    class_grid = np.full((2, 2, 2), -1, dtype=np.int16)
    ttl_grid = np.zeros((2, 2, 2), dtype=np.int16)
    apply_object_hits(class_grid, ttl_grid, np.empty((0, 4), dtype=np.int32), ttl_frames=10)
    assert np.all(class_grid == -1)
    assert np.all(ttl_grid == 0)


def test_decay_object_grids_clears_class_once_ttl_expires():
    class_grid = np.full((2, 2, 2), -1, dtype=np.int16)
    ttl_grid = np.zeros((2, 2, 2), dtype=np.int16)
    class_grid[0, 0, 0] = 7
    ttl_grid[0, 0, 0] = 1

    decay_object_grids(class_grid, ttl_grid)

    assert ttl_grid[0, 0, 0] == 0
    assert class_grid[0, 0, 0] == -1


def test_decay_object_grids_keeps_class_while_ttl_remains():
    class_grid = np.full((2, 2, 2), -1, dtype=np.int16)
    ttl_grid = np.zeros((2, 2, 2), dtype=np.int16)
    class_grid[0, 0, 0] = 7
    ttl_grid[0, 0, 0] = 3

    decay_object_grids(class_grid, ttl_grid)

    assert ttl_grid[0, 0, 0] == 2
    assert class_grid[0, 0, 0] == 7


def _one_hot_row(cx, cy, w, h, class_scores):
    return [cx, cy, w, h] + list(class_scores)


def test_decode_yolo_output_single_box():
    raw = np.array([_one_hot_row(100, 100, 20, 20, [0.9, 0.05, 0.05])], dtype=np.float32).T[None, :, :]
    detections = decode_yolo_output(raw, conf_threshold=0.4, iou_threshold=0.5, scale=1.0, pad=(0, 0))

    assert len(detections) == 1
    class_id, score, x1, y1, x2, y2 = detections[0]
    assert class_id == 0
    assert abs(score - 0.9) < 1e-4
    np.testing.assert_allclose([x1, y1, x2, y2], [90.0, 90.0, 110.0, 110.0], atol=1e-3)


def test_decode_yolo_output_nms_suppresses_duplicate_and_drops_low_confidence():
    rows = [
        _one_hot_row(100, 100, 20, 20, [0.9, 0.05, 0.05]),   # kept
        _one_hot_row(102, 101, 20, 20, [0.85, 0.1, 0.05]),   # overlaps the box above, suppressed
        _one_hot_row(300, 300, 10, 10, [0.1, 0.05, 0.05]),   # below confidence threshold
    ]
    raw = np.array(rows, dtype=np.float32).T[None, :, :]

    detections = decode_yolo_output(raw, conf_threshold=0.4, iou_threshold=0.5, scale=1.0, pad=(0, 0))

    assert len(detections) == 1
    class_id, score, *_ = detections[0]
    assert class_id == 0
    assert abs(score - 0.9) < 1e-4


def test_decode_yolo_output_maps_network_coords_back_to_original_image():
    raw = np.array([_one_hot_row(110, 60, 40, 20, [0.0, 0.9, 0.0])], dtype=np.float32).T[None, :, :]
    detections = decode_yolo_output(raw, conf_threshold=0.4, iou_threshold=0.5, scale=2.0, pad=(10, 10))

    assert len(detections) == 1
    class_id, score, x1, y1, x2, y2 = detections[0]
    assert class_id == 1
    np.testing.assert_allclose([x1, y1, x2, y2], [40.0, 20.0, 60.0, 30.0], atol=1e-3)


def test_coco_class_ids_resolves_known_names():
    assert coco_class_ids(("person",)) == (0,)
    assert coco_class_ids(("person", "chair")) == (0, COCO_CLASS_NAMES.index("chair"))
    assert coco_class_ids(()) == ()


def test_coco_class_ids_rejects_unknown_name():
    try:
        coco_class_ids(("person", "not_a_real_class"))
    except ValueError as exc:
        assert "not_a_real_class" in str(exc)
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_roll_object_grids_shifts_and_marks_exposed_slab_unknown()
    test_roll_object_grids_is_noop_without_movement()
    test_label_occupied_column_tags_existing_occupied_cells()
    test_label_occupied_column_empty_when_out_of_bounds_or_unoccupied()
    test_store_and_lookup_transform_direct_edge()
    test_lookup_transform_chains_multiple_hops()
    test_lookup_transform_same_frame_is_identity()
    test_lookup_transform_unresolved_returns_none()
    test_project_color_detections_to_voxels_identity_extrinsic()
    test_project_color_detections_to_voxels_applies_nonidentity_extrinsic()
    test_project_color_detections_to_voxels_handles_empty_and_out_of_frame()
    test_apply_object_hits_writes_class_and_refreshes_ttl()
    test_apply_object_hits_noop_on_empty_hits()
    test_decay_object_grids_clears_class_once_ttl_expires()
    test_decay_object_grids_keeps_class_while_ttl_remains()
    test_decode_yolo_output_single_box()
    test_decode_yolo_output_nms_suppresses_duplicate_and_drops_low_confidence()
    test_decode_yolo_output_maps_network_coords_back_to_original_image()
    test_coco_class_ids_resolves_known_names()
    test_coco_class_ids_rejects_unknown_name()
