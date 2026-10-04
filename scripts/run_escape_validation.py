"""Isolated live recovery validation; mission target is never replaced."""
import argparse
import copy
import fcntl
import json
import math
import os
import time
from pathlib import Path
from tool.simulator import ros_planning_web as web
from tool.simulator.recovery_strategies import proposals
from scripts.run_active_observation import evidence
from tool.simulator.observed_retreat_selection import shortlist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--duration', type=float, default=65)
    parser.add_argument('--snapshot', type=int, default=1)
    parser.add_argument('--side', choices=('left','right'), default='left')
    args = parser.parse_args()
    if os.getenv('ROS_DOMAIN_ID') != '217' or os.getenv('ROS_LOCALHOST_ONLY') != '1':
        raise ValueError('Dedicated local-only ROS domain 217 required')
    domain_lock = open('/tmp/tinynav-escape-domain-217.lock','w')
    fcntl.flock(domain_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = json.loads(Path('tinynav_temp/topology_pairs_20261003',f'snapshot-{args.snapshot}.json').read_text())
    saved = json.loads(Path('tinynav_temp/decision_observer', source['source_id']+'.json').read_text())
    state = source['evaluations']['v2']['request']['state']
    config = copy.deepcopy(saved['context']['scene_config'])
    xy = state['planning']['robot_world_xy']
    yaw = state['planning']['robot_yaw_deg']
    config['start'] = {'xy': xy, 'yaw_deg': yaw}
    frames = []
    web.start_ros()
    node = web.SIM_NODE
    started = time.monotonic()
    try:
        with node.lock:
            node.override = {'linear_mps': 0, 'yaw_radps': 0}
            node.override_until = time.monotonic()+600
        web.baseline_start(web.BaselineRequest(config=config, timeout_s=180))
        with node.lock:
            node.override = {'linear_mps': 0, 'yaw_radps': 0}
            node.override_until = time.monotonic()+600
            audit = source['source_audit']
            node.lab.cells = {(int(i),int(j)):v for i,j,v,_ in audit['cells']}
            now = time.monotonic()
            node.lab.cell_observed_at = {(int(i),int(j)):now-max(0,audit['captured_monotonic']-t) for i,j,_,t in audit['cells'] if t is not None}
        deadline = time.monotonic()+60
        while time.monotonic()<deadline:
            with node.lock:
                fresh = node.plan_received_at and time.monotonic()-node.plan_received_at < .5
            if fresh:
                break
            time.sleep(.2)
        if not fresh:
            raise RuntimeError('Planner did not publish a fresh report')
        total = 0
        previous = node.yaw_deg
        scan_start = time.monotonic()
        while total < 359.5 and time.monotonic()-scan_start < 25:
            with node.lock:
                node.override['yaw_radps'] = min(.6, math.radians(360-total)/.2)
            time.sleep(.125)
            with node.lock:
                total += abs((node.yaw_deg-previous+180)%360-180)
                previous = node.yaw_deg
        if total < 359.5 or node.collision or math.dist(xy,node.control_xy) > .02:
            raise RuntimeError('Active observation did not complete safely at anchor')
        with node.lock:
            node.override['yaw_radps'] = 0
        time.sleep(.5)
        with node.lock:
            plans = proposals(node.control_xy,node.yaw_deg,config['target'],config['robot'],node.lab.cells,node.lab.resolution,[])
            checked = evidence(plans,node.control_xy,node.yaw_deg,config['robot'],node.lab)
            screened = shortlist(plans,node.control_xy,node.yaw_deg,config['robot'],node.lab.cells,node.lab.resolution)
            eligible = screened['plans']
            if not eligible:
                raise RuntimeError('No fully observed long retreat offered')
            plan = next(p for p in eligible if p['id'].endswith(args.side))
            if time.monotonic()-node.plan_received_at>.5:
                raise RuntimeError('Fresh report required at recovery execution')
            node.recovery.start(plan,node.control_xy,node.yaw_deg,config['target'],time.monotonic())
        execution_start = time.monotonic()
        released = False
        while time.monotonic()-execution_start < args.duration:
            with node.lock:
                if not node.recovery.active and not released:
                    node.override_until = 0
                    released = True
                frames.append({'elapsed_s':time.monotonic()-execution_start,'xy':list(node.control_xy),'yaw_deg':node.yaw_deg,'recovery_active':bool(node.recovery.active),'collision':bool(node.collision),'report_age_s':time.monotonic()-node.plan_received_at})
                if not node.running:
                    break
            time.sleep(.25)
        with node.lock:
            result = {'kind':'supervised_observed_retreat_then_original_planner','model_used':False,'mission_target':config['target'],'scan_deg':total,'candidate_evidence':checked,'selection_rule':screened['rule'],'source_id':source['source_id'],'selected_strategy':plan,'events':node.recovery.events,'metrics':node.lab.metrics(),'frames':frames,'duration_s':time.monotonic()-started}
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ('frames','candidate_evidence')}),flush=True)
    finally:
        with node.lock:
            node.running = False
        web.shutdown()


if __name__ == '__main__':
    main()
