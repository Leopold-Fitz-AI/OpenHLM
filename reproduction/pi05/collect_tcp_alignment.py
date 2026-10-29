"""Collect VLA tests and clearly separated calibration/replay diagnostics."""
import json,csv
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]
protocols={
 'diagnostic':['g1_tcp_diagnostic01','g1_tcp_visualfit01','g1_tcp_floorfit_hold01','g1_tcp_matched50_vla01'],
 'fitted75_standardized':[f'g1_tcp_final_vla{i:02d}' for i in range(1,4)],
 'nominal58_standardized':[f'g1_tcp_fov58_vla{i:02d}' for i in range(1,3)],
 'privileged_calibration_not_vla':[f'g1_tcp_calibration{i:02d}' for i in range(1,6)],
 'recorded_action_replay_not_vla':['g1_tcp_push_replay01']}
results=[]
for protocol,names in protocols.items():
 for name in names:
    p=root/'logs'/name
    if not (p/'simulation.json').exists():raise RuntimeError(f'Incomplete {name}')
    s=json.loads((p/'simulation.json').read_text());c=json.loads((p/'client.json').read_text())
    rows=[x for x in json.loads((p/'task_telemetry.json').read_text()) if x['evaluating']]
    item={'name':name,'protocol':protocol,'mode':c['mode'],'evaluated_s':c['elapsed_seconds'],
       'steps':c['executed_steps'],'control_hz':c['effective_hz'],'transport_hz':c.get('transport_actual_hz'),
       'camera_hz':s.get('camera_actual_hz'),'fallen':s['fallen'],'task':s['task'],
       'standardized_start':c.get('standardized_start',False),'arm_bias_compensation':c.get('arm_bias_compensation',False)}
    if rows:
       close=[r for r in rows if r['gripper_commands']['rh']<.5]
       distance=lambda r:float(np.linalg.norm(r['tcp_diagnostics']['rh']['bottle_in_tcp_frame']))
       item.update({'min_right_command':min(r['gripper_commands']['rh'] for r in rows),
          'closest_tcp_to_bottle_center_m':min(map(distance,rows)),
          'closest_during_close_m':min(map(distance,close),default=None),
          'max_finger_force_N':max(max(v) for r in rows for v in r['finger_contact_forces'].values()),
          'max_tool_mount_error_m':max(g['mount_error_m'] for r in rows for g in r['tcp_diagnostics'].values())})
    if (p/'initial_state.json').exists():item['initial_root_qpos']=json.loads((p/'initial_state.json').read_text())['qpos'][:7]
    if (p/'camera_fit.json').exists():item['camera_fovy']={k:v['fovy'] for k,v in json.loads((p/'camera_fit.json').read_text())['cameras'].items()}
    if (p/'calibration_status.json').exists():item['last_calibration_status']=json.loads((p/'calibration_status.json').read_text())
    results.append(item)
summary={'runs':results,'protocol_scores':{k:{'attempts':len(v),'successes':sum(r['task']['success'] for r in results if r['protocol']==k)} for k,v in protocols.items()},
 'caveats':['manual jaw landmark fit is not recovered factory hand-eye calibration',
            '75 and 58 degree priors fit the same coplanar landmarks; actual stream intrinsics are required',
            'privileged calibration and action replay are not official VLA evaluation',
            'starting XY placement occurs before evaluation and leaves objects unchanged']}
out=root/'logs/tcp_alignment_delivery';out.mkdir(exist_ok=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary['protocol_scores'],indent=2))
for r in results:
 if r['protocol'].endswith('standardized'):print(r['name'],r['task'],r.get('min_right_command'),r.get('closest_tcp_to_bottle_center_m'),r.get('closest_during_close_m'))
