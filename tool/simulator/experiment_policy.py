"""Guards for short model actions in isolated ROS web simulations."""
import math


def guarded_command(record, report, xy, yaw_deg, generation, now_unix, report_age_s):
    if record['status'] != 'complete':return None, 'model_error'
    if record['context']['config_generation'] != generation:return None, 'scene_changed'
    if now_unix-record['created_at_unix'] > 8:return None, 'expired_result'
    if report is None or report_age_s > .5:return None, 'stale_plan'
    original=record['context']['planning_report']
    if math.dist(xy,original['robot_world_xy'])>.2 or abs((yaw_deg-original['robot_yaw_deg']+180)%360-180)>15:
        return None,'pose_changed'
    choice=record['response']['answers']['alternative']['choice']
    if not choice.startswith('candidate_'):return None,choice
    if choice not in record['request']['questions']['alternative']['criteria']:return None,'not_offered'
    ident=int(choice.removeprefix('candidate_'))
    candidate=next((c for c in report['candidates'] if c['id']==ident),None)
    previous=next((c for c in original['candidates'] if c['id']==ident),None)
    if not candidate or candidate['cost'] is None:return None,'collision_rejected'
    if not previous or abs(candidate['linear_mps']-previous['linear_mps'])>.1 or abs(candidate['angular_radps']-previous['angular_radps'])>.05:
        return None,'candidate_changed'
    if 'sampled_footprint_collision' in candidate['reasons']:return None,'collision_rejected'
    return {'candidate_id':ident,'linear_mps':candidate['control_linear_mps'], 'yaw_radps':candidate['control_yaw_radps'], 'duration_s':.75}, 'applied'
