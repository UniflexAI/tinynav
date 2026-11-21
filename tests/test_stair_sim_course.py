"""Check stacked floor dimensions independently of navigation results."""
import pytest

from tool.stair_sim_course import add_floor_stack, shaft_bounds, SHAFT, MULTI_SHAFT


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('landing_depth', [1.2, 1.75])
def test_ten_levels_have_eighteen_full_width_flights(direction, landing_depth):
    boxes = {}

    def box(name, x0, x1, y0, y1, bottom, top, *args, **kwargs):
        boxes[name] = (x0, x1, y0, y1, bottom, top)

    add_floor_stack(box, 18, direction, {}, lane_width=1.2, landing_depth=landing_depth)
    for flight in range(18):
        tread = boxes[f'flight_{flight}_tread_0']
        assert tread[3] - tread[2] == pytest.approx(1.2)
        assert tread[5] == pytest.approx(direction * (flight * 1.44 + .16))
        landing = boxes[f'landing_{flight}']
        assert landing[1] - landing[0] == pytest.approx(landing_depth)
        assert landing[3] - landing[2] == pytest.approx(2.54)
        assert landing[5] == pytest.approx(direction * (flight + 1) * 1.44)
    shaft = shaft_bounds(18, 1.2, landing_depth)
    assert shaft == pytest.approx((0., 2.4, .6, .74))
    assert boxes['flight_0_tread_0'][3] == pytest.approx(shaft[2])
    assert boxes['flight_1_tread_0'][2] == pytest.approx(shaft[3])


def test_default_shaft_geometry_is_unchanged():
    assert shaft_bounds(2) == SHAFT
    assert shaft_bounds(4) == MULTI_SHAFT
    assert shaft_bounds(2, landing_depth=1.2) == MULTI_SHAFT


@pytest.mark.parametrize('width', [0., .79, 3.01, float('nan')])
def test_invalid_flight_width_is_rejected(width):
    with pytest.raises(ValueError):
        shaft_bounds(18, width)
