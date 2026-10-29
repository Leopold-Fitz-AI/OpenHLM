"""Offline FK audit; privileged simulator data never enters VLA observations."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args();p=args.run_dir
m=mujoco.MjModel.from_xml_path(str(p/'scene.xml'));d=mujoco.MjData(m)
physics=np.load(p/'physics_trace.npz');control=np.load(p/('transport_trace.npz' if (p/'transport_trace.npz').exists() else 'control_trace.npz'));c=json.loads((p/'client.json').read_text())
precise='monotonic_time_s' in physics and 'monotonic_time_s' in control
idx=np.where(physics['times'][:,2]>0)[0];start=physics['times'][idx[0],0]
parts=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']
right=m.jnt_qposadr[[m.joint('right_'+part+'_joint').id for part in parts]]
left=m.jnt_qposadr[[m.joint('left_'+part+'_joint').id for part in parts]]
waist=m.jnt_qposadr[[m.joint('waist_'+part+'_joint').id for part in ['yaw','roll','pitch']]]
site=m.site('rh_tcp').id;bottle=m.body('task_bottle').id
rows=[]
for i in idx[::3]:
 t=float(physics['times'][i,0]-start)
 # Older recordings lack per-tick timestamps; allow <=~2 tick alignment error.
 j=int(np.clip(np.searchsorted(control['monotonic_time_s'],physics['monotonic_time_s'][i],side='right')-1,0,len(control['actions'])-1)) if precise else int(np.clip((t-.34)*len(control['actions'])/(c['elapsed_seconds']-.34),0,len(control['actions'])-1))
 a=control['actions'][j];d.qpos[:]=physics['qpos'][i];mujoco.mj_forward(m,d)
 actual=d.site_xpos[site].copy();object_pos=d.xpos[bottle].copy()
 d.qpos[right]=a[8:15];d.qpos[left]=a[:7];d.qpos[waist]=a[28:31];mujoco.mj_forward(m,d);target=d.site_xpos[site].copy()
 rows.append({'time_s':t,'gripper_command':float(a[15]),'actual_tcp':actual.tolist(),'command_fk_tcp':target.tolist(),
              'actual_distance_m':float(np.linalg.norm(actual-object_pos)),
              'command_fk_distance_m':float(np.linalg.norm(target-object_pos)),
              'command_actual_tcp_error_m':float(np.linalg.norm(target-actual))})
close=[x for x in rows if x['gripper_command']<.5]
summary={'note':'FK of emitted upper-body reference at measured base/legs; native response lag remains', 'timestamp_alignment':'monotonic timestamps' if precise else 'approximate for old recording',
         'minimum_actual_tcp_distance_m':min(x['actual_distance_m'] for x in rows),
         'minimum_command_fk_distance_m':min(x['command_fk_distance_m'] for x in rows),
         'mean_command_actual_tcp_error_m':float(np.mean([x['command_actual_tcp_error_m'] for x in rows])),
         'closest_during_closure':min(close,key=lambda x:x['actual_distance_m']) if close else None,
         'closest_command_target':min(rows,key=lambda x:x['command_fk_distance_m'])}
(p/'tcp_tracking_summary.json').write_text(json.dumps(summary,indent=2));(p/'tcp_tracking.json').write_text(json.dumps(rows));print(json.dumps(summary,indent=2))
