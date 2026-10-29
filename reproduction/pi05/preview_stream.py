"""Publish genuine future reference windows to the original SONIC stream.

Reference frames carry a 50 Hz time grid. A new stream epoch uses the native
catch-up reset once at initialization; body preview does not require delaying
current reference commands until future samples have arrived.
"""
import threading,time
import numpy as np
from scipy.spatial.transform import Rotation
from sonic_g1_env import pack_pose_message,_REAL_TO_ISAAC_IDX

class ProbeTrajectory:
    def __init__(self,default):self.default=default.copy();self.epoch=time.monotonic()
    def window(self,stamps):
        t=np.maximum(0,np.asarray(stamps)-self.epoch);out=np.tile(self.default,(len(t),1));vel=np.zeros_like(out)
        for phase,joint in [(0,8),(1,11),(2,13)]:
            mask=(t>=phase*12)&(t<(phase+1)*12);s=t[mask]-phase*12
            out[mask,joint]+=.12*np.sin(2*np.pi*s/6)
            vel[mask,joint]=.12*2*np.pi/6*np.cos(2*np.pi*s/6)
        return out,vel

class PreviewWindowStream:
    def __init__(self,env,trajectory,hz=50,lookahead_s=.2):
        self.env=env;self.trajectory=trajectory;self.hz=hz;self.count=round(lookahead_s*hz)+1
        self.epoch=time.monotonic();self.base_frame=env._frame_index+1000
        self.stop_event=threading.Event();self.errors=[];self.actions=[];self.times=[]
        self.frame_starts=[];self.thread=threading.Thread(target=self.run,name='sonic-future-reference',daemon=True)
    def start(self):self.thread.start()
    def submit(self,action):pass # the trajectory owns future references, not instantaneous targets
    def run(self):
        deadline=time.monotonic()
        try:
            while not self.stop_event.is_set():
                now=time.monotonic();offset=max(0,int((now-self.epoch)*self.hz))
                frames=np.arange(offset,offset+self.count,dtype=np.int64)
                stamps=self.epoch+frames/self.hz
                if hasattr(self.trajectory,'sample'):
                    a,v,quat=self.trajectory.sample(stamps)
                else:
                    a,v=self.trajectory.window(stamps)
                    if np.max(abs(a[:,33]))>1e-7:raise ValueError('Fixed-yaw preview baseline received yaw velocity')
                    quat=Rotation.from_euler('xyz',np.c_[a[:,31:33],np.zeros(len(a))]).as_quat()[:,[3,0,1,2]]
                joints=np.concatenate([a[:,16:31],a[:,:7],a[:,8:15]],1)
                velocity=np.concatenate([v[:,16:31],v[:,:7],v[:,8:15]],1)
                packet={'joint_pos':joints[:,_REAL_TO_ISAAC_IDX].astype(np.float32),
                        'joint_vel':velocity[:,_REAL_TO_ISAAC_IDX].astype(np.float32),
                        'body_quat_w':quat.astype(np.float32),'frame_index':frames+self.base_frame,
                        'action_hand_left':a[0,7:8].astype(np.float32),
                        'action_hand_right':a[0,15:16].astype(np.float32)}
                self.env._zmq_socket.send(pack_pose_message(packet,topic='pose',version=1))
                self.actions.append(a[0].copy());self.times.append(now);self.frame_starts.append(int(frames[0]))
                deadline+=1/self.hz
                if deadline<time.monotonic()-1/self.hz:deadline=time.monotonic()
                self.stop_event.wait(max(0,deadline-time.monotonic()))
        except Exception as exc:self.errors.append(exc)
    def check(self):
        if self.errors:raise RuntimeError('Preview reference sender failed') from self.errors[0]
    def close(self):self.stop_event.set();self.thread.join(timeout=5);self.check()
