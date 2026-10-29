"""Privileged G1 TCP/contact calibration reference. NOT a VLA policy.

Uses simulator state only to isolate mechanics/kinematics from perception. Sends
ordinary joint targets through original SonicG1Env/SONIC; no object attachment.
"""
import json,sys
from pathlib import Path
import numpy as np
# Use the simulator's MuJoCo version for its XML, without changing training deps.
sim_packages=Path(__file__).resolve().parents[2]/'.venv_sim/lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
sys.path.insert(0,str(sim_packages))
import mujoco
sys.path.pop(0)
from scipy.optimize import least_squares
from arm_tracking import ArmBiasCompensator

class G1TCPCalibration:
    def __init__(self,run,default,status):
        self.run=run;self.default=default.copy()
        self.model=mujoco.MjModel.from_xml_path(str(run/'scene.xml'));self.data=mujoco.MjData(self.model)
        parts=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']
        self.ids=[self.model.joint('right_'+p+'_joint').id for p in parts]
        self.qidx=self.model.jnt_qposadr[self.ids];self.bounds=self.model.jnt_range[self.ids]
        self.site=self.model.site('rh_tcp').id
        self.bottle=np.array(status['task']['bottle_position']);self.grasp=self.bottle+[-.02,0,.04]
        spec=json.loads((run/'task_spec.json').read_text())
        self.bin_target=np.array([spec['bin_center'][0]-.10,spec['bin_center'][1]+.10,1.04])
        self.approach_x=None
        self.stage='pregrasp';self.stage_start=0.;self.previous=None;self.integral=np.zeros(3)
        self.tracker=ArmBiasCompensator(indices=list(range(8,15)))
        self.command=default[8:15].copy();self.events=[];self.last_solve_t=-1.;self.target0=None
    def transition(self,stage,t,tcp):
        self.events.append({'stage':self.stage,'finished_s':t,'tcp':tcp.tolist()})
        self.stage=stage;self.stage_start=t;self.target0=tcp.copy()
        (self.run/'calibration_stages.json').write_text(json.dumps(self.events,indent=2))
    def action(self,status,t):
        self.data.qpos[:]=status['qpos'];mujoco.mj_forward(self.model,self.data)
        tcp=self.data.site_xpos[self.site].copy();tau=t-self.stage_start
        if self.target0 is None:self.target0=tcp.copy()
        score=status['task'];grip=1.
        goals={'pregrasp':self.grasp+[-.12,0,0],'approach':self.grasp,
               'close':self.grasp,'lift':self.grasp+[0,0,.20],
               'transfer':self.bin_target,'release':self.bin_target,
               'withdraw':self.bin_target+[0,0,.12],'settle':self.bin_target+[0,0,.12]}
        duration={'pregrasp':8.,'approach':6.,'close':1.,'lift':5.,'transfer':10.,'release':1.,'withdraw':3.,'settle':1.}
        goal=goals[self.stage];a=min(1,tau/duration[self.stage]);a=a*a*(3-2*a)
        target=(1-a)*self.target0+a*goal
        if self.stage=='approach':
            if self.approach_x is None:self.approach_x=float(self.target0[0])
            normal=self.data.site_xmat[self.site].reshape(3,3)[:,1]
            aligned=abs(tcp[1]-goal[1])<.008 and abs(tcp[2]-goal[2])<.015 and normal[1]>.985
            if aligned:self.approach_x=min(float(goal[0]),self.approach_x+.012/30)
            elif abs(tcp[1]-goal[1])>.015 and tcp[0]>self.bottle[0]-.09:
                self.approach_x=min(self.approach_x,float(goal[0]-.10))
            target=np.array([self.approach_x,goal[1],goal[2]])
        error=goal-tcp
        if self.stage in ['lift','transfer']:grip=0.
        elif self.stage=='close':grip=max(0,1-tau)
        elif self.stage=='release':grip=min(1,tau)
        if self.stage=='pregrasp' and tau>8 and np.linalg.norm(error)<.018:self.transition('approach',t,tcp);return self.action(status,t)
        elif self.stage=='approach' and tau>6 and abs(error[0])<.012 and abs(error[1])<.008 and abs(error[2])<.025:self.transition('close',t,tcp);return self.action(status,t)
        elif self.stage=='close' and score['grasped']:self.transition('lift',t,tcp);return self.action(status,t)
        elif self.stage=='lift' and tau>5 and score['lifted']:self.transition('transfer',t,tcp);return self.action(status,t)
        elif self.stage=='transfer' and tau>10 and np.linalg.norm(error)<.018:self.transition('release',t,tcp);return self.action(status,t)
        elif self.stage=='release' and tau>2:self.transition('withdraw',t,tcp);return self.action(status,t)
        elif self.stage=='withdraw' and tau>3:self.transition('settle',t,tcp);return self.action(status,t)
        if self.stage=='close' and tau>7:raise RuntimeError('Calibration failed to establish bilateral grasp')
        if self.stage in ['pregrasp','approach'] and tau>22:raise RuntimeError('Calibration TCP failed to converge')
        if t-self.last_solve_t>=.05:
            dt=max(.02,t-self.last_solve_t) if self.last_solve_t>=0 else .05
            self.integral=np.clip(self.integral+(target-tcp)*dt*.35,-.04,.04)
            desired=target+self.integral
            neutral=self.data.qpos[self.qidx].copy()
            def residual(q):
                self.data.qpos[self.qidx]=q;mujoco.mj_forward(self.model,self.data)
                normal=self.data.site_xmat[self.site].reshape(3,3)[:,1]
                return np.r_[(self.data.site_xpos[self.site]-desired)*30,
                             np.cross(normal,[0,1,0])*.7,max(0,1-normal[1])*.2,(q-neutral)*.02]
            sol=least_squares(residual,np.clip(neutral,self.bounds[:,0]+1e-5,self.bounds[:,1]-1e-5),
                              bounds=(self.bounds[:,0]+1e-5,self.bounds[:,1]-1e-5),max_nfev=35)
            proposed=self.default.copy();proposed[8:15]=sol.x
            measured=self.default.copy();measured[8:15]=neutral
            corrected=self.tracker.update(proposed,measured,dt)
            goal_q=np.clip(corrected[8:15],self.bounds[:,0]+1e-4,self.bounds[:,1]-1e-4)
            self.command+=np.clip(goal_q-self.command,-.9*dt,.9*dt)
            self.last_solve_t=t
        action=self.default.copy();action[8:15]=self.command;action[15]=grip
        (self.run/'calibration_status.json').write_text(json.dumps({'stage':self.stage,'time_s':t,'tcp':tcp.tolist(),
            'goal':goal.tolist(),'tcp_error_m':float(np.linalg.norm(error)),'grip':grip,'privileged_not_vla':True}))
        return action
