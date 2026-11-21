from types import SimpleNamespace

import pytest

from tool.stair_sim_control import apply_drive_command


def command(vx, wz):
    return SimpleNamespace(linear=SimpleNamespace(x=vx), angular=SimpleNamespace(z=wz))


@pytest.mark.parametrize('pause_tick', [1, 2, 3, 4])
def test_resume_reapplies_published_command_before_next_navigation_callback(pause_tick):
    published = command(.25, -.3)
    drive = SimpleNamespace(vx=0., wz=0.)
    observed = []
    for tick in range(10):
        if tick == pause_tick:
            drive.vx = drive.wz = 0.  # The web pause freezes physics and clears drive.
        apply_drive_command(drive, published)
        observed.append((drive.vx, drive.wz))
    assert observed == [(.25, -.3)] * 10


def test_resume_obeys_a_newly_published_stop_instead_of_old_motion():
    drive = SimpleNamespace(vx=.25, wz=-.3)
    apply_drive_command(drive, command(0., 0.))
    assert (drive.vx, drive.wz) == (0., 0.)
    apply_drive_command(drive, None)
    assert (drive.vx, drive.wz) == (0., 0.)
