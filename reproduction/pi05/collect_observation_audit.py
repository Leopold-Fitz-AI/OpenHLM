"""Summarize final repeated trials without reclassifying unsuccessful attempts."""
import json
from pathlib import Path
import csv
import numpy as np
root=Path(__file__).resolve().parents[2]
trials=[]
for i in range(1,4):
    name=f'g1_bottle_worker{i:02d}';p=root/'logs'/name
    s=json.loads((p/'simulation.json').read_text());c=json.loads((p/'client.json').read_text())
    assert s['evaluation_started'] and c['executed_steps']>0, name
    r=[x for x in json.loads((p/'task_telemetry.json').read_text()) if x['evaluating']]
    telemetry=list(csv.DictReader((p/'telemetry.csv').open()))
    tr={'name':name,'evaluation_seconds':c['elapsed_seconds'],'control_hz':c['effective_hz'],
        'steps':c['executed_steps'],'policy_queries':c['policy_queries'],
        'camera_hz':s['camera_actual_hz'],'fallen':s['fallen'],'task':s['task'],
        'right_command_min':min(x['gripper_commands']['rh'] for x in r),
        'right_measured_opening_min':min(x['gripper_openings']['rh'] for x in r),
        'max_finger_force_N':max(max(v) for x in r for v in x['finger_contact_forces'].values()),
        'max_tool_force_N':max(v for x in r for v in x['tool_contact_forces'].values()),
        'max_receive_age_seconds':c['max_receive_age_seconds'],
        'max_physics_backlog_s':s['max_physics_backlog_s'],
        'simulation_to_wall_time_ratio':(float(telemetry[-1]['sim_time'])-float(telemetry[0]['sim_time']))/(float(telemetry[-1]['wall_time'])-float(telemetry[0]['wall_time'])),
        'policy_latency_mean_s':float(np.mean(c['policy_seconds'])),
        'policy_latency_p95_s':float(np.percentile(c['policy_seconds'],95)),
        'final_bottle_position':r[-1]['bottle_position']}
    if (p/'control_trace.npz').exists():
        a=np.load(p/'control_trace.npz');error=np.abs(a['actions'][:,8:15]-a['measured_states'][:,8:15])
        tr['right_arm_same_tick_command_measurement_mae_rad']=error.mean(axis=0).tolist()
        tr['tracking_metric_note']='includes response lag; not a static calibration error'
    trials.append(tr)
summary={'trials':trials,'grasped':sum(x['task']['grasped'] for x in trials),
         'lifted':sum(x['task']['lifted'] for x in trials),'successes':sum(x['task']['success'] for x in trials),
         'attempts':len(trials),'excluded_before_evaluation':['g1_bottle_final_audit_01'],
         'scope':'official VLA + original SONIC, approximate cameras and physical gripper surrogate; no object attachment',
         'known_mismatches':['uncalibrated camera intrinsics/extrinsics','surrogate CTAG geometry and opening calibration','synthetic appearance','camera capture below 30Hz','receiver timestamps do not align sensor exposure']}
out=root/'logs/vla_observation_audit/final_summary.json';out.write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
