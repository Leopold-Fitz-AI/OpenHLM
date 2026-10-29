"""Bounded joint-reference bias using only measured robot joints, no objects."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

class ArmBiasCompensator:
    def __init__(self,indices=None,gain=.3,limit=.35):
        self.indices=np.array(indices if indices is not None else [*range(7),*range(8,15)])
        self.gain=gain;self.limit=limit;self.bias=np.zeros(34)
        root=Path(__file__).resolve().parents[2]
        robot=ET.parse(root/'src/GR00T-WholeBodyControl4OpenHLM/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml')
        self.lower=np.full(34,-np.inf);self.upper=np.full(34,np.inf)
        parts=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']
        for side,offset in [('left',0),('right',8)]:
            for i,part in enumerate(parts):
                lo,hi=map(float,robot.find(f".//joint[@name='{side}_{part}_joint']").get('range').split())
                self.lower[offset+i]=lo;self.upper[offset+i]=hi
    def update(self,target,measured,dt):
        target=np.asarray(target);measured=np.asarray(measured)
        if target.shape!=(34,) or measured.shape!=(34,) or not np.isfinite([target,measured]).all():
            raise ValueError('Invalid arm tracking input')
        idx=self.indices
        self.bias[idx]=np.clip(self.bias[idx]+self.gain*(target[idx]-measured[idx])*min(dt,.1),-self.limit,self.limit)
        out=target.copy();out[idx]+=self.bias[idx]
        out[idx]=np.clip(out[idx],self.lower[idx],self.upper[idx])
        return out
