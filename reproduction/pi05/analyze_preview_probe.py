"""Separate transport preview, learned tracking and physical motor response."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args();run=args.run_dir
u=np.load(run/'transport_trace.npz');d=np.load(run/'wbc_debug_trace.npz');start=u['monotonic_time_s'][0]
t=u['monotonic_time_s']-start;td=d['monotonic_time_s']-start;q=d['q']
issued=np.concatenate([u['actions'][:,16:31],u['actions'][:,:7],u['actions'][:,8:15]],1)
def fitlag(tx,x,ty,y):
 best=None
 for lag in np.arange(0,1.001,.005):
  valid=(ty-lag>=tx[0])&(ty-lag<=tx[-1]);yy=y[valid];xx=np.interp(ty[valid]-lag,tx,x)
  A=np.c_[xx,np.ones_like(xx)];gain,bias=np.linalg.lstsq(A,yy,rcond=None)[0]
  mse=np.mean((A@np.array([gain,bias])-yy)**2)
  if best is None or mse<best['mse']:best={'lag_s':float(lag),'gain':float(gain),'bias_rad':float(bias),'mse':float(mse)}
 return best
results=[]
for stage,joint,label in [(0,22,'right_shoulder_pitch'),(1,25,'right_elbow'),(2,27,'right_wrist_pitch')]:
 mask=(td>stage*12+1)&(td<(stage+1)*12-.3);tt=td[mask]
 result={'joint':label,'issued_to_reference':fitlag(t,issued[:,joint],tt,q[mask,0,joint]),
         'reference_to_motor':fitlag(td,q[:,0,joint],tt,q[mask,1,joint]),
         'motor_to_measured':fitlag(td,q[:,1,joint],tt,q[mask,2,joint]),
         'reference_minus_measured_mean_rad':float(np.mean(q[mask,0,joint]-q[mask,2,joint])),
         'motor_minus_measured_mean_rad':float(np.mean(q[mask,1,joint]-q[mask,2,joint]))}
 results.append(result)
summary={'method':'sinusoidal least squares gain/bias/lag fit; sampled diagnostics have ~33ms resolution','results':results,
         'per_joint_reference_minus_motor_mean':np.mean(q[:,0]-q[:,1],0).tolist(),
         'per_joint_motor_minus_measured_mean':np.mean(q[:,1]-q[:,2],0).tolist()}
(run/'preview_analysis.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
