"""Strict completion gate for a scripted G1 pick-and-bin trial."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();r=args.run
sim=json.loads((r/'simulation.json').read_text());client=json.loads((r/'client.json').read_text());status=json.loads((r/'status.json').read_text());spec=json.loads((r/'task_spec.json').read_text());cfg=json.loads((r/'controller_config.json').read_text());task=status['task'];errors=[]
def require(test,message):
    if not test:errors.append(message)
require(client['mode']=='scripted-baseline' and client['executed_steps']>0,'Not an executed scripted G1 trial')
require(all(sim['task'][k] for k in ['grasped','lifted','success']),'Grasp/lift/placement criteria did not all pass')
require(not sim['fallen'] and sim['finite_state'] and sim['physics_worker_error'] is None,'Physical stability or real-time check failed')
require(status['height']>.6 and status['tilt_radians']<.4,'Final robot pose is not stable')
lo,hi=task['bottle_low'],task['bottle_high']
inside=all(lo[i]>=spec['bin_center'][i]-spec['bin_half_size'][i]-.003 and hi[i]<=spec['bin_center'][i]+spec['bin_half_size'][i]+.003 for i in (0,1))
inside=inside and lo[2]>=spec['bin_floor']-.015 and hi[2]<=spec['bin_rim']-.01
require(inside,'Bottle is not fully in bin at final sample')
require(math.sqrt(sum(v*v for v in task['bottle_velocity'][3:]))<.15,'Bottle is moving at final sample')
require(task['gripper_openings']['rh']>.85 and task['gripper_commands']['rh']>.85,'Right gripper is not released')
require(max(task['tool_contact_forces'].values())<.1,'Bottle remains in contact with gripper')
placement=json.loads((r/'initial_placement.json').read_text())
require(placement['before_evaluation'] and not placement['objects_moved'],'Invalid initialization placement')
result={'run':r.name,'passed':not errors,'errors':errors,'not_vla':True,'right_arm_reference_servo':cfg.get('right_arm_reference_servo',False),'task':sim['task'],'evaluation_seconds':client['elapsed_seconds'],'final_height_m':status['height'],'final_bottle_position':task['bottle_position'],'physics_step_s':sim.get('physics_step_s'),'camera_hz':sim['camera_actual_hz']}
(r/'baseline_gate.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));raise SystemExit(bool(errors))
