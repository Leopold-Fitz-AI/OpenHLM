"""Time-preserving VLA chunk preview; no simulator or object-state access."""
from dataclasses import dataclass
import time
import numpy as np
from scipy.spatial.transform import Rotation

@dataclass(frozen=True)
class ChunkPlan:
    actions:np.ndarray
    start:float
    yaw0:float

class ActionChunkTrajectory:
    def __init__(self,default,initial_yaw=0.,hz=30,execute_steps=25):
        self.hz=hz;self.limit=execute_steps
        self.plan=ChunkPlan(np.tile(default,(50,1)),time.monotonic(),initial_yaw)
    def _yaw(self,plan,stamps):
        rate=plan.actions[:self.limit,33]
        prefix=np.r_[0,np.cumsum((rate[:-1]+rate[1:])/(2*self.hz))]
        u=np.clip((np.asarray(stamps)-plan.start)*self.hz,0,self.limit)
        i=np.minimum(np.floor(u).astype(int),self.limit-1);f=u-i
        following=rate[np.minimum(i+1,self.limit-1)]
        return plan.yaw0+prefix[i]+(rate[i]*f+.5*(following-rate[i])*f*f)/self.hz
    def replace(self,actions,start=None):
        a=np.array(actions,dtype=np.float64,copy=True)
        if a.shape!=(50,34) or not np.isfinite(a).all():raise ValueError('Expected finite 50x34 official action chunk')
        stamp=time.monotonic() if start is None else start
        yaw=float(self._yaw(self.plan,[stamp])[0]);a.setflags(write=False)
        self.plan=ChunkPlan(a,stamp,yaw)
    def index(self):return max(0,int((time.monotonic()-self.plan.start)*self.hz))
    def sample(self,stamps):
        plan=self.plan;t=np.maximum(0,np.asarray(stamps)-plan.start)
        u=np.minimum(t*self.hz,self.limit-1)
        u=np.where(abs(u-np.rint(u))<1e-9,np.rint(u),u)
        i=np.floor(u).astype(int);j=np.minimum(i+1,self.limit-1);f=u-i
        a=plan.actions[i]*(1-f[:,None])+plan.actions[j]*f[:,None]
        v=(plan.actions[j]-plan.actions[i])*self.hz
        a[:,[7,15]]=plan.actions[i][:,[7,15]]
        a[t>=self.limit/self.hz,33]=0.
        euler=np.c_[a[:,31:33],self._yaw(plan,stamps)]
        quat=Rotation.from_euler('xyz',euler).as_quat()[:,[3,0,1,2]]
        return a,v,quat
    def window(self,stamps):
        a,v,_=self.sample(stamps);return a,v
