"""compute_path_climb labelling: what must read as climbing, and what must not.

A miss costs strictness (the planner keeps the tight threshold there), so these
pin the shapes that MUST be labelled as much as the ones that must not."""
import numpy as np

from tinynav.core.path_climb import (
    PathClimbIndex, bake, compute_path_climb, n_climbing)

SPACING = 0.1  # capture samples every ~10 cm of travel


def _poses(z, spacing=SPACING):
    """Straight walk along +x at `spacing`, with the given z profile."""
    return {i: np.array([[1, 0, 0, i * spacing],
                         [0, 1, 0, 0.0],
                         [0, 0, 1, zi],
                         [0, 0, 0, 1.0]]) for i, zi in enumerate(z)}


def _climb_span(z):
    """(min_x, max_x) of the samples labelled climbing, or None."""
    out = compute_path_climb(_poses(z))
    lab = out[:, 3] >= 0.5
    return (out[lab, 0].min(), out[lab, 0].max()) if lab.any() else None


def _ramp(x, start, length, rise):
    return np.clip((x - start) / length, 0.0, 1.0) * rise


def test_single_riser_is_climbing():
    # One 0.17 m step at x=3, taken over 0.3 m of travel: the regression case
    # (a small platform the robot refused to step onto).
    x = np.arange(0, 6, SPACING)
    span = _climb_span(_ramp(x, 3.0, 0.3, 0.17))
    assert span is not None
    # lead-in ~= win_m minus the riser's horizontal extent, so the hint is up
    # well before the step (and stays up past it)
    assert span[0] <= 2.4 and span[1] >= 3.8


def test_full_flight_is_climbing():
    x = np.arange(0, 6, SPACING)
    span = _climb_span(_ramp(x, 2.0, 3.0, 0.85))
    assert span is not None and span[0] <= 2.0 and span[1] >= 5.0


def test_flat_walk_with_gait_bob_is_not_climbing():
    # +/-5 cm bob at a ~0.6 m stride: large steps, but the sign vote splits and
    # the net over the window is ~0.
    x = np.arange(0, 6, SPACING)
    assert _climb_span(0.05 * np.sin(2 * np.pi * x / 0.6)) is None


def test_bob_on_top_of_a_riser_still_climbs():
    x = np.arange(0, 6, SPACING)
    z = _ramp(x, 3.0, 0.3, 0.17) + 0.03 * np.sin(2 * np.pi * x / 0.6)
    assert _climb_span(z) is not None


def test_vio_teleport_is_not_climbing():
    x = np.arange(0, 6, SPACING)
    z = np.where(x >= 3.0, 1.0, 0.0)  # 1 m jump between consecutive samples
    assert _climb_span(z) is None


def test_descending_is_climbing_too():
    # The hint gates the obstacle z-span filter; going down a step needs it as much.
    x = np.arange(0, 6, SPACING)
    assert _climb_span(-_ramp(x, 3.0, 0.3, 0.17)) is not None


def test_index_lookup_off_path_is_flat():
    x = np.arange(0, 6, SPACING)
    idx = PathClimbIndex(compute_path_climb(_poses(_ramp(x, 3.0, 0.3, 0.17))))
    assert idx.on_stairs([3.0, 0.0, 0.17])
    assert not idx.on_stairs([3.0, 9.0, 0.17])  # beyond the association radius


def test_short_path_is_flat():
    assert compute_path_climb({})[:, 3].size == 0
    assert not compute_path_climb(_poses([0.0, 0.2]))[:, 3].any()


# --- the region form: what the publisher actually sends ---------------------

def test_climbing_within_returns_only_climbing_samples():
    x = np.arange(0, 6, SPACING)
    idx = PathClimbIndex(compute_path_climb(_poses(_ramp(x, 3.0, 0.3, 0.17))))
    pts = idx.climbing_within([3.0, 0.0, 0.17], radius_m=10.0)
    assert len(pts) and pts.shape[1] == 3
    # every returned sample is one the labeller marked climbing
    span = _climb_span(_ramp(x, 3.0, 0.3, 0.17))
    assert pts[:, 0].min() >= span[0] - 1e-6 and pts[:, 0].max() <= span[1] + 1e-6


def test_climbing_within_is_bounded_by_the_radius():
    x = np.arange(0, 6, SPACING)
    idx = PathClimbIndex(compute_path_climb(_poses(_ramp(x, 3.0, 0.3, 0.17))))
    near = idx.climbing_within([3.0, 0.0, 0.17], radius_m=0.5)
    assert len(near) and np.abs(near[:, 0] - 3.0).max() <= 0.5 + 1e-6
    # unlike nearest_value, the radius here is the caller's, not assoc_m
    assert len(near) < len(idx.climbing_within([3.0, 0.0, 0.17], radius_m=10.0))


def test_climbing_within_far_away_is_empty():
    x = np.arange(0, 6, SPACING)
    idx = PathClimbIndex(compute_path_climb(_poses(_ramp(x, 3.0, 0.3, 0.17))))
    assert len(idx.climbing_within([3.0, 50.0, 0.17], radius_m=3.5)) == 0


def test_flat_capture_yields_no_region():
    x = np.arange(0, 6, SPACING)
    idx = PathClimbIndex(compute_path_climb(_poses(0.05 * np.sin(2 * np.pi * x / 0.6))))
    assert len(idx.climbing_within([3.0, 0.0, 0.0], radius_m=10.0)) == 0


def test_bake_writes_labels_and_is_idempotent(tmp_path):
    x = np.arange(0, 6, SPACING)
    np.save(tmp_path / 'poses.npy', _poses(_ramp(x, 3.0, 0.3, 0.17)), allow_pickle=True)
    assert 'samples climbing' in bake(str(tmp_path))
    assert (tmp_path / 'path_climb.npy').exists()
    assert 'already baked' in bake(str(tmp_path))
    assert 'samples climbing' in bake(str(tmp_path), force=True)


def test_n_climbing_counts_the_label_column():
    x = np.arange(0, 6, SPACING)
    labels = compute_path_climb(_poses(_ramp(x, 3.0, 0.3, 0.17)))
    assert n_climbing(labels) == int((labels[:, 3] >= 0.5).sum()) > 0
    assert n_climbing(compute_path_climb({})) == 0


def test_bake_on_a_map_without_poses_is_not_an_error(tmp_path):
    assert 'no poses.npy' in bake(str(tmp_path))
    assert not (tmp_path / 'path_climb.npy').exists()
