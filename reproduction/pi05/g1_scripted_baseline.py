"""Settled waypoint baseline through original SONIC, with real contact scoring.

Privileged object/robot state is permitted ONLY in this scripted baseline.
No object weld, no force assistance, no resets after evaluation begins.
"""
from dataclasses import dataclass
from collections import deque
import json,sys,time
from pathlib import Path
import numpy as np
packages=Path(__file__).resolve().parents[2]/'.venv_sim/lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
sys.path.insert(0,str(packages));import mujoco;sys.path.pop(0)
from scipy.optimize import least_squares,lsq_linear
from scipy.spatial.transform import Rotation

@dataclass(frozen=True)
class JointPlan:
    start:float
    duration:float
    q0:np.ndarray
    q1:np.ndarray
    def sample(self,stamps):
        s=np.clip((np.asarray(stamps)-self.start)/self.duration,0,1)
        blend=10*s**3-15*s**4+6*s**5
        rate=(30*s**2-60*s**3+30*s**4)/self.duration
        return self.q0[None]+blend[:,None]*(self.q1-self.q0)[None],rate[:,None]*(self.q1-self.q0)[None]

class G1ScriptedBaseline:
    def __init__(self,run,default,status):
        self.run=run;self.default=default.copy();self.hand=1.;self.epoch=time.monotonic()
        self.model=mujoco.MjModel.from_xml_path(str(run/'scene.xml'));self.data=mujoco.MjData(self.model)
        parts=['shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']
        names=['right_'+p+'_joint' for p in parts]
        self.ids=np.array([self.model.joint(n).id for n in names]);self.qidx=self.model.jnt_qposadr[self.ids]
        self.dofs=self.model.jnt_dofadr[self.ids];self.action_idx=np.arange(8,15)
        self.bounds=self.model.jnt_range[self.ids].copy()
        # Keep the upper arm outside the torso. Joint-range-only IK can
        # otherwise solve a precise TCP pose by passing through the chest.
        self.bounds[1]=[-1.4,-.15]
        torso_ids={self.model.body(n).id for n in ['torso_link','pelvis']}
        arm_ids={self.model.body('right_'+n+'_link').id for n in ['shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw']}
        torso_geoms=[g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in torso_ids and self.model.geom_contype[g]]
        arm_geoms=[g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in arm_ids and self.model.geom_contype[g]]
        self.collision_pairs=[(a,b) for a in arm_geoms for b in torso_geoms]
        self.site=self.model.site('rh_tcp').id;self.tool=self.model.body('rh_tool').id
        self.bottle=np.array(status['task']['bottle_position']);b=self.bottle
        spec=json.loads((run/'task_spec.json').read_text())
        self.goals={'raise':np.array([.24,-.28,.97]),'pre_high':b+[-.15,0,.18],
                    'pre_low':b+[-.15,0,.02],'insert':b+[-.005,0,.02],
                    'lift':b+[-.02,0,.22], 'transfer_near':np.array([.31,-.28,1.06]),
                    'transfer_bin':np.array([spec['bin_center'][0]-.10,spec['bin_center'][1]+.10,1.06]),
                    'withdraw':np.array([spec['bin_center'][0]-.10,spec['bin_center'][1]+.10,1.18])}
        self.order=['raise','pre_high','pre_low','insert','grip','lift','transfer_near','transfer_bin','release','withdraw','settle']
        self.stage='raise';self.stage_start=self.epoch;self.new_stage=True;self.hold_R=None
        self.plan=JointPlan(self.epoch,1.,default.copy(),default.copy())
        self.events=[];self.corrections=0;self.settled_since=None;self.last_ref=default.copy()
        self.tcp_history=deque()
        self.grip_local=None;self.contact_lost_since=None
        self.insert_x=float(self.goals['pre_low'][0]);self.goal=self.goals['raise'].copy()
    def window(self,stamps):
        plan=self.plan;q,v=plan.sample(stamps);q[:,15]=self.hand;q[:,7]=1.
        return q,v
    def set_joint_goal(self,q,seconds):
        now=time.monotonic();q0=self.plan.sample([now])[0][0]
        largest=np.max(abs(q[self.action_idx]-q0[self.action_idx]))
        duration=max(seconds,largest/.28,1.)
        self.plan=JointPlan(now,duration,q0,q.copy());self.last_ref=q.copy()
        self.settled_since=None
    def rotation_error(self,R):
        if self.hold_R is not None:return Rotation.from_matrix(self.hold_R@R.T).as_rotvec()
        return np.cross(R[:,1],[0,1,0])
    def ik(self,goal):
        neutral=self.data.qpos[self.qidx].copy()
        def residual(q):
            self.data.qpos[self.qidx]=q;mujoco.mj_forward(self.model,self.data)
            R=self.data.site_xmat[self.site].reshape(3,3)
            return np.r_[(self.data.site_xpos[self.site]-goal)*30,self.rotation_error(R)*5,
                         (q-neutral)*.12,
                         [max(0,.008-mujoco.mj_geomDistance(self.model,self.data,a,b,.05,None))*40 for a,b in self.collision_pairs]]
        candidates=[]
        outward_seed=neutral.copy();outward_seed[1]=-.4;outward_seed[2]=.4
        for seed in [neutral,self.last_ref[self.action_idx],outward_seed]:
            sol=least_squares(residual,np.clip(seed,self.bounds[:,0]+1e-4,self.bounds[:,1]-1e-4),
                    bounds=(self.bounds[:,0]+1e-4,self.bounds[:,1]-1e-4),max_nfev=100)
            candidates.append((sol.cost,sol.x.copy()))
        q=min(candidates,key=lambda x:x[0])[1];out=self.last_ref.copy()
        # Retain settled tracking compensation when advancing a waypoint.
        # IK solves a change from measured joints; it is not a replacement
        # for the reference that produced those measured joints.
        out[self.action_idx]=np.clip(self.last_ref[self.action_idx]+q-neutral,
            self.bounds[:,0]+1e-4,self.bounds[:,1]-1e-4)
        return out
    def next_stage(self,now,tcp):
        self.events.append({'stage':self.stage,'end_s':now-self.epoch,'tcp':tcp.tolist(),'corrections':self.corrections})
        self.stage=self.order[self.order.index(self.stage)+1];self.stage_start=now;self.new_stage=True
        self.corrections=0;self.settled_since=None
        (self.run/'baseline_stages.json').write_text(json.dumps(self.events,indent=2))
    def action(self,status,t):
        now=time.monotonic();self.data.qpos[:]=status['qpos'];mujoco.mj_forward(self.model,self.data)
        tcp=np.array(status['task']['tcp_diagnostics']['rh']['actual_tcp'])
        R=self.data.xmat[self.tool].reshape(3,3).copy();score=status['task'];tau=now-self.stage_start
        self.tcp_history.append((now,tcp.copy()))
        while self.tcp_history[0][0]<now-.4:self.tcp_history.popleft()
        mean_tcp=np.mean([p for _,p in self.tcp_history],axis=0)
        if status['height']<.60 or status['tilt_radians']>.45:raise RuntimeError('Baseline lost standing stability')
        if self.stage in ['raise','pre_high','pre_low','insert'] and np.linalg.norm(np.array(score['bottle_position'])-self.bottle)>.04:
            raise RuntimeError('Bottle left the grasp approach window')
        if self.stage in ['lift','transfer_near','transfer_bin']:
            if min(score['finger_contact_forces']['rh'])<.2:
                if self.contact_lost_since is None:self.contact_lost_since=now
                if now-self.contact_lost_since>.5:raise RuntimeError('Grasp lost bilateral contact')
            else:self.contact_lost_since=None
            if self.grip_local is not None and np.linalg.norm(np.array(score['tcp_diagnostics']['rh']['bottle_in_tcp_frame'])-self.grip_local)>.03:
                raise RuntimeError('Bottle slipped more than 3cm in gripper')
        if self.new_stage:
            self.new_stage=False
            if self.stage=='pre_low':self.hold_R=Rotation.from_euler('y',.25).as_matrix()
            if self.stage in self.goals:
                self.goal=self.goals[self.stage].copy()
                if self.stage=='insert':self.goal[0]=self.insert_x
                if self.stage in ['lift','transfer_near','transfer_bin']:
                    delta=self.goal-tcp;self.goal=tcp+delta*min(1,.04/np.linalg.norm(delta))
                self.set_joint_goal(self.ik(self.goal),6. if self.stage in ['raise','pre_high','lift','transfer_near','transfer_bin'] else 3.)
        if self.stage=='grip':
            self.hand=max(0,1-tau/2)
            if score['grasped'] and tau>3 and min(score['finger_contact_forces']['rh'])>1:
                self.hold_R=R.copy();self.goals['lift']=tcp+[0,0,.18]
                self.grip_local=np.array(score['tcp_diagnostics']['rh']['bottle_in_tcp_frame'])
                self.next_stage(now,tcp)
            elif tau>6:raise RuntimeError('Bilateral grasp was not established')
        elif self.stage=='release':
            self.hand=min(1,tau/2)
            if tau>3:self.next_stage(now,tcp)
        elif self.stage=='settle':
            self.hand=1.
            if tau>12 and not score['success']:raise RuntimeError('Released bottle did not settle in bin')
        else:
            self.hand=0. if self.stage in ['lift','transfer_near','transfer_bin'] else 1.
            error=self.goal-mean_tcp;normal_error=np.linalg.norm(self.rotation_error(R))
            ready=(now>self.plan.start+self.plan.duration+1.)
            tolerance=.012 if self.stage in ['insert','transfer_bin'] else .018
            aligned=np.linalg.norm(error)<tolerance and normal_error<.16
            if self.stage in ['pre_low','insert']:
                aligned=abs(error[1])<.006 and abs(error[2])<.015 and abs(error[0])<.014 and normal_error<.13
                aligned=aligned and abs(self.goal[1]-tcp[1])<.009
            if ready and aligned:
                if self.settled_since is None:self.settled_since=now
                if now-self.settled_since>.4:
                    if self.stage=='insert' and self.insert_x<self.goals['insert'][0]-.002:
                        self.insert_x=min(self.goals['insert'][0],self.insert_x+.010)
                        self.goal=self.goals['insert'].copy();self.goal[0]=self.insert_x
                        self.set_joint_goal(self.ik(self.goal),2.0)
                    elif self.stage in ['lift','transfer_near','transfer_bin'] and np.linalg.norm(self.goals[self.stage]-self.goal)>.005:
                        delta=self.goals[self.stage]-self.goal
                        self.goal=self.goal+delta*min(1,.04/np.linalg.norm(delta))
                        self.set_joint_goal(self.ik(self.goal),1.5)
                    elif self.stage=='lift' and not score['lifted']:
                        if tau>25:raise RuntimeError('Lift contact did not persist')
                    else:self.next_stage(now,tcp)
            elif ready:
                self.settled_since=None
                # Corrections are applied only after the preview and motor
                # response have settled, avoiding a fast integral wind-up loop.
                self.data.qpos[:]=status['qpos'];mujoco.mj_forward(self.model,self.data)
                jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
                mujoco.mj_jacSite(self.model,self.data,jp,jr,self.site)
                A=np.vstack([jp[:,self.dofs],jr[:,self.dofs]*.06])
                desired=np.r_[error,self.rotation_error(R)*.06]*.5
                ref=self.last_ref.copy()
                lower=np.maximum(self.bounds[:,0]+1e-4-ref[self.action_idx],-.035)
                upper=np.minimum(self.bounds[:,1]-1e-4-ref[self.action_idx],.035)
                # Solve within the limits. Clipping an unconstrained inverse
                # can remove its main positive motion while retaining the
                # compensating negative motions, moving away from the goal.
                dq=lsq_linear(np.vstack([A,np.eye(len(self.ids))*np.sqrt(.00015)]),
                    np.r_[desired,np.zeros(len(self.ids))],bounds=(lower,upper),tol=1e-7).x
                ref[self.action_idx]=np.clip(ref[self.action_idx]+dq,
                    self.bounds[:,0]+1e-4,self.bounds[:,1]-1e-4)
                self.set_joint_goal(ref,1.0);self.corrections+=1
            if not self.new_stage and tau>(180 if self.stage=='insert' else 80):
                raise RuntimeError(f'Baseline stage {self.stage} failed to converge')
        self.data.qpos[:]=status['qpos'];mujoco.mj_forward(self.model,self.data)
        q=self.window([now])[0][0]
        record={'stage':self.stage,'time_s':now-self.epoch,'tcp':tcp.tolist(),'goal':self.goal.tolist(),
                'position_error_m':float(np.linalg.norm(self.goal-tcp)),
                'normal_error_rad':float(np.linalg.norm(self.rotation_error(R))),
                'corrections':self.corrections,'hand_command':self.hand,'not_vla':True}
        (self.run/'baseline_status.tmp').write_text(json.dumps(record));(self.run/'baseline_status.tmp').replace(self.run/'baseline_status.json')
        return q
