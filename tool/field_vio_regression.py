"""Stream real bag timing through the VIO supervisor and map/control reset callbacks.

Does not replay topics or construct a robot backend/SDK. Cache markers explicitly
exercise invalidation; this is not full perception replanning or physical replay.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import rclpy
from rclpy.parameter import Parameter
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import Bool
from tinynav.core import stair_node, planning_node, vio_safety_node
from tinynav.platforms import cmd_vel_control


def run(root):
    results=[]
    rclpy.init()
    now=0.
    clock=SimpleNamespace(monotonic=lambda:now)
    for module in (stair_node,planning_node,cmd_vel_control,vio_safety_node):module.time=clock
    for bag in sorted(root.iterdir()):
        dbs=sorted(bag.glob('*.db3')) if bag.is_dir() else []
        if not dbs:continue
        guard=vio_safety_node.VioSafetyNode(source_clock=lambda:t0+now)
        stair=stair_node.StairNode(SimpleNamespace(camera_height=.45,direction='up',floors=4,landings_per_floor=2,turn='auto',memory=None,rate=2.))
        planner=planning_node.PlanningNode();control=cmd_vel_control.CmdVelControlNode()
        control._on_nav_active(Bool(data=True));control.cmd_pub=Mock()
        planner.path_pub=Mock();planner.height_map_pub=Mock();planner.obstacle_mask_pub=Mock()
        stair.poi_change_pub=Mock();stair.status_pub=Mock()
        events=[];last_epoch=None;old_locked=False;last_fault=0;blocked_checks=0;lock_checks=0;seeded=False;allow_release=False
        def state(msg):
            nonlocal last_epoch,old_locked,last_fault,seeded
            s=json.loads(msg.data);epoch=(s['session'],s['epoch'])
            control.vio_safety_callback(msg);planner.vio_safety_callback(msg);stair.vio_safety_callback(msg)
            if epoch!=last_epoch:
                assert not np.any(planner.occupancy_grid) and not stair.gen.frames
                assert planner.target_pose is None and control.path is None
                events.append(dict(time=now,epoch=s['epoch'],locked=s['locked'],fault_id=s['fault_id'],reason=s['reason'],cache_markers_present=seeded))
                seeded=False;last_epoch=epoch
            if old_locked and not s['locked']:
                assert not s['blocked'] and s['stable_for'] >= guard.guard.settle_s
                assert epoch != last_release[0], 'Recovery did not change map epoch'
                last_release[0] = epoch
            old_locked=s['locked'];last_fault=s['fault_id']
            if not s['blocked']:
                # Represent cached pre-fault data without pretending to recompute measured depth maps.
                planner.occupancy_grid[0,0,0]=.2
                planner.target_pose=np.array([1.,0.,0.])
                stair.gen.frames=deque([(now,np.array([[0.,0.,0.]]))])
                control.path=object();seeded=True
        last_release=[None]
        guard.state_pub=SimpleNamespace(publish=state)
        count=0;t0=None
        for db in dbs:
            with sqlite3.connect(f'file:{db}?mode=ro',uri=True) as c:
                metadata={i:(name,get_message(typ)) for i,name,typ in c.execute('select id,name,type from topics') if name in ('/slam/odometry','/slam/depth','/camera/camera/vio_status','/nav/paused')}
                ids=list(metadata);marks=','.join('?' for _ in ids)
                if t0 is None:t0=c.execute('select min(timestamp) from messages').fetchone()[0]/1e9
                for timestamp,topic,blob in c.execute(f'select timestamp,topic_id,data from messages where topic_id in ({marks}) order by timestamp',ids):
                    now=timestamp/1e9-t0;guard.publish_state()
                    name,typ=metadata[topic];msg=deserialize_message(blob,typ);count+=1
                    if name=='/slam/odometry':guard.pose_callback(msg)
                    elif name=='/slam/depth':guard.depth_callback(msg)
                    elif name=='/camera/camera/vio_status':guard.status_callback(msg)
                    elif name=='/nav/paused':stair.pause_callback(msg);control._on_paused(msg)
                    s=guard.publish_state();control.cmd_timer_callback()
                    if s['blocked']:
                        command=control.cmd_pub.publish.call_args.args[0]
                        assert command.linear.x==command.linear.y==command.angular.z==0.
                        assert not np.any(planner.occupancy_grid) and not stair.gen.frames
                        blocked_checks+=1
                    if s['locked']:lock_checks+=1
        s=guard.publish_state()
        result=dict(bag=bag.name,messages=count,events=events,blocked_zero_checks=blocked_checks,
                    latched_checks=lock_checks,final_state=s,passed=True)
        if s['locked']:
            response=guard.confirm_parameters([Parameter('confirm_fault_id',value=s['fault_id'])])
            assert not response.successful, 'Legacy confirmation bypassed automatic settling'
            result['legacy_confirmation_rejected']=True

        results.append(result)
        for node in (guard,stair,planner,control):node.destroy_node()
    rclpy.shutdown()
    assert len(results)==5, f'Expected five real bags, got {len(results)}'
    return dict(passed=True,scope=__doc__,bags=results)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bags',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    result=run(args.bags);args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
