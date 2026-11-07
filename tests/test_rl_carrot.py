import torch

from tinynav.rl.sim import GEO_UNREACHABLE, NavSim, SimConfig


def scene(carrot=2.2):
    sim = NavSim(SimConfig(num_envs=1, n_boxes=1, carrot_distance=carrot), device="cpu")
    sim.boxes.zero_()
    sim.pos[0] = torch.tensor([-3.0, 0.0])
    sim.goal[0] = torch.tensor([3.0, 0.0])
    sim._geodesic(torch.tensor([0]))
    return sim


def test_fixed_goal_unchanged():
    sim = scene(0)
    assert torch.equal(sim.policy_goal(), sim.goal)


def test_carrot_moves_and_preserves_destination():
    sim = scene()
    goal = sim.goal.clone()
    target = sim.policy_goal()
    assert 1.9 < (target - sim.pos).norm() < 2.5
    sim.pos[:, 0] += 0.8
    assert sim.policy_goal()[0, 0] > target[0, 0] + 0.5
    assert torch.equal(sim.goal, goal)
    sim.pos[0] = torch.tensor([2.0, 0.0])
    assert torch.equal(sim.policy_goal(), goal)


def test_carrot_routes_around_wall():
    sim = scene()
    sim.boxes[0, 0] = torch.tensor([-2.0, 0.0, 0.1, 1.5, 1.0])
    sim._geodesic(torch.tensor([0]))
    target = sim.policy_goal()
    assert abs(target[0, 1]) > 1.0
    assert sim._geo_at(torch.tensor([0]), target).item() < GEO_UNREACHABLE
    assert sim._point_clear(sim.boxes, target, 0.0).all()
