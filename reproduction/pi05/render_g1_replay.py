"""Render recorded G1 physics; never advances or changes a simulated state."""
import argparse,json,subprocess
from pathlib import Path
import cv2,mujoco,numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=float,default=0);p.add_argument('--fps',type=float,default=10);args=p.parse_args();run=args.run
cv2.setNumThreads(1)
z=np.load(run/'physics_trace.npz');records=json.loads((run/'task_telemetry.json').read_text());valid=z['times'][:,2].astype(bool);epoch=z['monotonic_time_s'][np.flatnonzero(valid)[0]];times=z['monotonic_time_s']-epoch
m=mujoco.MjModel.from_xml_path(str(run/'scene.xml'));d=mujoco.MjData(m)
m.vis.global_.offwidth=1280;m.vis.global_.offheight=720
overview=mujoco.Renderer(m,height=720,width=960);head=mujoco.Renderer(m,height=200,width=232);wrist=mujoco.Renderer(m,height=240,width=320)
from fisheye_camera import FisheyeCamera
fish=FisheyeCamera(m,'head_camera') if json.loads((run/'simulation.json').read_text())['camera_profile']=='approximate-fisheye' else None
cam=mujoco.MjvCamera();cam.lookat[:]=[.38,-.23,.68];cam.distance=2.1;cam.azimuth=140;cam.elevation=-22
args.output.parent.mkdir(parents=True,exist_ok=True)
ff=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24','-s','1280x720','-r',str(args.fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)],stdin=subprocess.PIPE)
frames=0
try:
    for t in np.arange(args.start,times[-1],1/args.fps):
        k=int(np.argmin(abs(times-t)));d.qpos[:]=z['qpos'][k];d.qvel[:]=z['qvel'][k];d.time=z['times'][k,1];mujoco.mj_forward(m,d)
        overview.update_scene(d,camera=cam);left=cv2.cvtColor(overview.render(),cv2.COLOR_RGB2BGR)
        panels=[]
        for name in ['head_camera','eval_left_wrist','eval_right_wrist']:
            if name=='head_camera' and fish:im=fish.render(d)
            else:
                r=head if name=='head_camera' else wrist;r.update_scene(d,camera=name);im=r.render().copy()
            im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR);h,w=im.shape[:2];scale=min(320/w,240/h);small=cv2.resize(im,(round(w*scale),round(h*scale)));panel=np.zeros((240,320,3),np.uint8);y=(240-small.shape[0])//2;x=(320-small.shape[1])//2;panel[y:y+small.shape[0],x:x+small.shape[1]]=small
            cv2.putText(panel,name.upper(),(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA);panels.append(panel)
        cv2.rectangle(left,(0,0),(960,62),(20,20,20),-1)
        cv2.putText(left,'G1 SCRIPT + RIGHT ARM SERVO / NOT VLA',(16,25),cv2.FONT_HERSHEY_SIMPLEX,.63,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(left,f'RECORDED PHYSICS REPLAY | 1x | t={times[k]:.1f}s',(16,49),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
        score=records[k];label='   '.join(f'{key.upper()} {int(score.get(key,False))}' for key in ['grasped','lifted','success'])
        cv2.rectangle(left,(0,680),(960,720),(20,20,20),-1);cv2.putText(left,label,(16,705),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),1,cv2.LINE_AA)
        frame=np.concatenate([left,np.concatenate(panels,axis=0)],axis=1);ff.stdin.write(frame.tobytes());frames+=1
finally:
    ff.stdin.close();ff.wait();overview.close();head.close();wrist.close()
    if fish:fish.close()
if ff.returncode:raise RuntimeError('Video encoding failed')
print(json.dumps({'frames':frames,'fps':args.fps,'replay':True,'state_interpolation':False,'output':str(args.output)}))
