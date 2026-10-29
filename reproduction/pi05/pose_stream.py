"""Timestamped 30 Hz pose targets resampled to the original 50 Hz WBC feed."""
from collections import deque
import threading
import time
import numpy as np


def interpolate_target(samples, stamp):
    if stamp <= samples[0][0]: return samples[0][1].copy()
    for (ta,a),(tb,b) in zip(samples,samples[1:]):
        if ta <= stamp <= tb:
            weight=(stamp-ta)/max(tb-ta,1e-9)
            out=(1-weight)*a+weight*b
            # Teleop hand commands are discrete, not blended joint positions.
            out[[7,15]]=a[[7,15]] if stamp < tb else b[[7,15]]
            return out
    return samples[-1][1].copy()


class TimedPoseStream:
    def __init__(self,send,initial,hz=50,source_hz=30):
        self.send=send;self.hz=hz;self.delay=1/source_hz
        self.samples=deque([(time.monotonic()-self.delay,np.asarray(initial).copy())],maxlen=8)
        self.lock=threading.Lock();self.stop_event=threading.Event();self.errors=[]
        self.actions=[];self.times=[]
        self.thread=threading.Thread(target=self.run,name='wbc-pose-stream',daemon=True)
    def start(self):self.thread.start()
    def submit(self,action):
        with self.lock:self.samples.append((time.monotonic(),np.asarray(action).copy()))
    def run(self):
        deadline=time.monotonic()
        try:
            while not self.stop_event.is_set():
                now=time.monotonic()
                with self.lock:action=interpolate_target(list(self.samples),now-self.delay)
                self.send(action);self.times.append(now);self.actions.append(action.copy())
                deadline+=1/self.hz
                if deadline < time.monotonic()-1/self.hz:deadline=time.monotonic()
                self.stop_event.wait(max(0,deadline-time.monotonic()))
        except Exception as exc:self.errors.append(exc)
    def check(self):
        if self.errors:raise RuntimeError('WBC pose sender failed') from self.errors[0]
    def close(self):self.stop_event.set();self.thread.join(timeout=5);self.check()
