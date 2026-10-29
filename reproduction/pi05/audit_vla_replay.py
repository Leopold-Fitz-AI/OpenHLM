"""Fixed-noise official demonstration replay and crossed image/state ablations."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from openpi.policies import policy_config
from openpi.training import config
from openpi_client import image_tools

root = Path(__file__).resolve().parents[2]
out = root / 'logs/vla_observation_audit'
out.mkdir(parents=True, exist_ok=True)
episode = root / 'public_data/example_demonstration/20260520_1602_example_1/episode_0000'
raw = json.loads((episode / 'data.json').read_text())
rows = raw['data']
keys = ['head_image_left', 'left_wrist_image', 'right_wrist_image']
fields = ['rgb_left', 'wrist_rgb_left', 'wrist_rgb_right']
def vector(row, kind):
    b = np.asarray(row[kind + '_body'], dtype=np.float32)
    return np.concatenate([b[18:25], [row[kind+'_hand_left']], b[25:32],
                           [row[kind+'_hand_right']], b[3:18], b[:3]]).astype(np.float32)
def demo(index):
    row=rows[index]
    obs={'state':vector(row,'state')}
    panels=[]
    for key, field in zip(keys, fields):
        ref=row[field]; cap=cv2.VideoCapture(str(episode/ref['video_path']))
        cap.set(cv2.CAP_PROP_POS_FRAMES,ref['frame_index']); ok, bgr=cap.read();cap.release()
        assert ok
        obs[key]=image_tools.resize_with_pad(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),224,224)
        panels.append(cv2.cvtColor(obs[key],cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out/f'demo_{index}.jpg'),np.concatenate(panels,1))
    return obs
sim=dict(np.load(root/'logs/g1_bottle_vla_01/first_observation.npz'))
cv2.imwrite(str(out/'sim_before.jpg'),np.concatenate([cv2.cvtColor(sim[k],cv2.COLOR_RGB2BGR) for k in keys],1))
torch.set_num_threads(16)
policy=policy_config.create_trained_policy(config.get_config('openhlm_example'),root/'models/12tasks_12full-teleop',pytorch_device='cuda',sample_kwargs={'num_steps':10})
noise=np.random.default_rng(42).standard_normal((50,34)).astype(np.float32)
prompt='Walk towards the platform and put the green bottle into the dust bin.'
results=[]
def run(name,obs,gt=None):
    request={f'observation/{k}':v for k,v in obs.items()};request['prompt']=prompt
    actions=policy.infer(request,noise=noise)['actions']
    np.save(out/f'{name}_actions.npy',actions)
    record={'name':name,'right_grip_min':float(actions[:,15].min()),'right_grip_first25_mean':float(actions[:25,15].mean()),'left_grip_min':float(actions[:,7].min()),'right_arm_first':actions[0,8:15].tolist(),'input_state':obs['state'].tolist()}
    if gt is not None:record['target_right_grip']=float(gt[15]);record['first_action_mae']=float(np.abs(actions[0]-gt).mean())
    results.append(record);print(json.dumps(record),flush=True)
    (out/'audit.json').write_text(json.dumps(results,indent=2))
for index in [0,350,380,400,600,1250,1300]:
    obs=demo(index);run(f'demo_{index}',obs,vector(rows[index],'action'))
    if index in [380,400,600]:
        run(f'demo_images_sim_state_{index}',{**obs,'state':sim['state']})
        run(f'sim_images_demo_state_{index}',{**sim,'state':obs['state']})
run('sim_original',sim)
