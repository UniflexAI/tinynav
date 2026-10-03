import copy
from scripts.run_active_observation import evidence
from tool.simulator.navigation_lab import NavigationLab
from tool.simulator.recovery_strategies import poses
from tool.simulator.decision_information import footprint_cells


def setup():
    robot = {'length': .4, 'width': .3}
    plan = {'id': 'probe', 'stages': [{'name': 'probe', 'linear_mps': .3, 'yaw_radps': 0, 'duration_s': 20}]}
    cells = set()
    for point, _ in poses([0, 0], 0, plan['stages']):
        cells.update(footprint_cells(point, robot, .1))
    lab = NavigationLab()
    lab.cells = {cell: 'clear' for cell in cells}
    return robot, plan, cells, lab


def test_exact_unknown_count_cannot_disappear_through_rounding():
    robot, plan, cells, lab = setup()
    cell = next(iter(cells))
    del lab.cells[cell]
    row = evidence([plan], [0, 0], 0, robot, lab)['probe']
    assert row['stages'][0]['unknown_cells'] == 1
    assert round(row['stages'][0]['unknown_fraction'], 2) == 0
    assert not row['all_stages_exactly_observed_clear']


def test_measured_blocking_and_missing_returns_do_not_count_as_clear():
    robot, plan, cells, lab = setup()
    lab.cells[next(iter(cells))] = 'blocked'
    row = evidence([plan], [0, 0], 0, robot, lab)['probe']
    assert row['stages'][0]['blocked_cells'] == 1
    assert not row['all_stages_exactly_observed_clear']
    lab.cells.clear()
    row = evidence([plan], [0, 0], 0, robot, lab)['probe']
    assert row['stages'][0]['unknown_cells'] == row['stages'][0]['cells']


def test_observation_audit_does_not_mutate_commands_or_cells():
    robot, plan, cells, lab = setup()
    original = copy.deepcopy(plan)
    occupancy = dict(lab.cells)
    assert evidence([plan], [0, 0], 0, robot, lab)['probe']['all_stages_exactly_observed_clear']
    assert plan == original and lab.cells == occupancy
