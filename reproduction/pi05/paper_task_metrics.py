"""Four equally weighted rubric components from OpenHLM Appendix A, task 3.

Dwell times, pose tolerances and contact thresholds operationalize prose and
remain proxy assumptions until calibrated against real evaluation footage.
"""
from dataclasses import dataclass,field
import math

@dataclass
class PaperDisposalScore:
    mechanics:object
    home_xy:tuple
    approach_xy:tuple
    home_tolerance:float=.18
    approach_tolerance:float=.20
    approached:bool=False
    returned:bool=False
    pedal_opened:bool=False
    milestones:dict=field(default_factory=dict)
    _approach_since:float|None=None
    _home_since:float|None=None
    _pedal_since:float|None=None
    initial_yaw:float|None=None

    @property
    def grasped(self):return self.mechanics.grasped
    @property
    def lifted(self):return self.mechanics.lifted
    @property
    def success(self):return all(self.points().values())
    def points(self):
        return {'approach_and_return':int(self.approached and self.returned),
                'green_bottle_grasp':int(self.mechanics.grasped),
                'left_foot_pedal_open':int(self.pedal_opened),
                'bottle_inside_bin':int(self.mechanics.success)}
    def update(self,t,root_xy,yaw,root_speed,left_foot_force,lid_angle,pedal_displacement):
        if not all(math.isfinite(v) for v in [t,*root_xy,yaw,root_speed,left_foot_force,lid_angle,pedal_displacement]):
            raise ValueError('Non-finite paper rubric observation')
        if self.initial_yaw is None:self.initial_yaw=yaw
        near=math.dist(root_xy,self.approach_xy)<self.approach_tolerance
        self._approach_since=t if near and self._approach_since is None else self._approach_since if near else None
        if self._approach_since is not None and t-self._approach_since>=.4 and not self.approached:
            self.approached=True;self.milestones['approached_bin']=t
        home=math.dist(root_xy,self.home_xy)<self.home_tolerance
        heading=abs(math.atan2(math.sin(yaw-self.initial_yaw),math.cos(yaw-self.initial_yaw)))<.35
        returned=self.approached and home and heading and root_speed<.25
        self._home_since=t if returned and self._home_since is None else self._home_since if returned else None
        if self._home_since is not None and t-self._home_since>=.5 and not self.returned:
            self.returned=True;self.milestones['returned_home']=t
        pressing=left_foot_force>5. and lid_angle>.9 and pedal_displacement<-.012
        self._pedal_since=t if pressing and self._pedal_since is None else self._pedal_since if pressing else None
        if self._pedal_since is not None and t-self._pedal_since>=.25 and not self.pedal_opened:
            self.pedal_opened=True;self.milestones['left_foot_opened_lid']=t
    def result(self):
        points=self.points();out=self.mechanics.result()
        out.update({'mechanics_success':self.mechanics.success,'success':self.success,
            'task_progress':sum(points.values())/4,'paper_points':points,'paper_points_total':sum(points.values()),
            'paper_points_max':4,'approached_bin':self.approached,'returned_home':self.returned,
            'left_foot_pedal_open':self.pedal_opened,'rubric_events_sim_time':dict(self.milestones),
            'evaluation_domain':'simulation_proxy_not_paper_real_world'})
        return out
