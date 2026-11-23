import json
import pytest
from tinynav.core.vio_safety import VioSafety, safety_blocked


def feed(g, t, p=(0.,0.,0.), status='TRACKING'):
    g.status(status,t)
    g.pose(1000+t,p,t)
    g.depth(1000+t,t)
    return g.snapshot(t)


def recover(g, start, p=(2.,0.,0.)):
    for i in range(26):state=feed(g,start+i*.1,p)
    return state


def test_meter_jump_auto_recovers_with_new_epoch_without_confirmation(tmp_path):
    g=VioSafety(tmp_path/'state.json');feed(g,0)
    s=feed(g,.1,(2.,0.,0.));assert s['locked'] and s['blocked']
    epoch=s['epoch'];assert not g.confirm(s['fault_id'],.2)[0]
    s=recover(g,.2)
    assert not s['blocked'] and not s['locked'] and s['epoch']>epoch
    assert s['recovery_mode']=='automatic' and not s['ready']
    assert not json.loads((tmp_path/'state.json').read_text())['locked']


@pytest.mark.parametrize('status',['RESTARTING','NOT_INITED','LOST'])
def test_status_recovery_alone_is_not_enough_but_fresh_inputs_resume(status):
    g=VioSafety();feed(g,0);g.status(status,.1)
    assert g.snapshot(.1)['locked']
    g.status('TRACKING',1.)
    assert g.snapshot(2.)['blocked']
    assert not recover(g,4.)['blocked']


def test_pose_dropout_stops_then_recovers_automatically():
    g=VioSafety();feed(g,0)
    s=g.snapshot(.51);assert s['locked'] and s['reason']=='pose_timeout'
    assert g.pose_rx is g.depth_rx is None
    assert not recover(g,1.)['blocked']


def test_depth_dropout_latches_even_with_fresh_odometry():
    g=VioSafety();feed(g,0)
    for t in [.2,.4,.6,.8]:g.pose(1000+t,(0,0,0),t);s=g.snapshot(t)
    assert s['locked'] and s['reason']=='depth_timeout'


def test_small_jump_clears_frame_without_hard_latch():
    g=VioSafety();feed(g,0);old=g.epoch
    s=feed(g,.01,(.2,0,0));assert g.epoch>old and s['blocked'] and not s['locked']
    assert not recover(g,.1,(.2,0,0))['blocked']


def test_lock_survives_supervisor_restart(tmp_path):
    path=tmp_path/'state.json';g=VioSafety(path);feed(g,0);feed(g,.1,(2,0,0))
    restored=VioSafety(path)
    assert restored.locked and restored.fault_id==g.fault_id
    assert not restored.snapshot(100.)['ready']
    assert not recover(restored,101.)['blocked']
    assert not restored.locked


def test_new_fault_invalidates_old_confirmation():
    g=VioSafety();feed(g,0);first=feed(g,.1,(2,0,0))['fault_id'];recover(g,.2)
    second=feed(g,2.8,(4,0,0))['fault_id'];assert second>first
    recover(g,2.9,(4,0,0));assert not g.confirm(first,5.4)[0]


def test_stale_duplicate_messages_do_not_keep_stream_alive():
    g=VioSafety();feed(g,0)
    for t in [.2,.4,.6]:g.pose(1000,(0,0,0),t);g.depth(1000,t)
    assert g.snapshot(.6)['locked']


def test_guard_heartbeat_expiry_blocks_consumers():
    assert safety_blocked({'blocked':False},0.,1.01)
    assert not safety_blocked({'blocked':False},0.,.5)


def test_restored_lock_requires_new_stable_interval(tmp_path):
    path=tmp_path/'state.json';g=VioSafety(path);feed(g,0);feed(g,.1,(2,0,0))
    restored=VioSafety(path)
    assert not feed(restored,100.,(2,0,0))['ready']
    assert not restored.confirm(restored.fault_id,100.)[0]


def test_supervisor_restart_also_waits_for_automatic_recovery(tmp_path):
    path=tmp_path/'state.json';g=VioSafety(path);feed(g,0.)
    restarted=VioSafety(path)
    assert restarted.locked and restarted.reason=='supervisor_restart'
    assert restarted.epoch>g.epoch
    assert not recover(restarted,2.)['blocked']


def test_old_publisher_cannot_release_new_session_fault():
    from tinynav.core.vio_safety import accept_safety
    current={'session':'new','epoch':3,'locked':True}
    assert not accept_safety({'session':'old','epoch':2,'blocked':False},current)
    assert not accept_safety({'session':'other','epoch':3,'blocked':False},current)
    assert accept_safety({'session':'new','epoch':3,'blocked':False},current)


def test_one_depth_frame_cannot_release_a_long_outage():
    g=VioSafety();feed(g,0)
    for i in range(1,42):
        t=i*.1;g.status('TRACKING',t);g.pose(1000+t,(0,0,0),t);g.snapshot(t)
    g.depth(1004.1,4.1)
    assert g.snapshot(4.1)['locked']
    assert not recover(g,4.2,(0,0,0))['blocked']


def test_stability_interval_restarts_after_another_gap():
    g=VioSafety();feed(g,0);g.fault('test',.1)
    for i in range(11):feed(g,.2+i*.1)
    assert g.snapshot(2.)['blocked']
    assert feed(g,2.1)['locked']
    assert not recover(g,2.2)['blocked']
