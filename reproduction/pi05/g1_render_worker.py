"""Camera publishing and video in a separate process from physical control."""
import argparse,json,struct,time,os
from pathlib import Path
import cv2,msgpack,mujoco,numpy as np,zmq
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--camera-profile',required=True);p.add_argument('--run-label',required=True);args=p.parse_args();run=args.run_dir
model=mujoco.MjModel.from_xml_path(str(run/'scene.xml'));data=mujoco.MjData(model)
head_renderer=mujoco.Renderer(model,height=200,width=232)
wrist_renderer=mujoco.Renderer(model,height=240,width=320)
renderer=mujoco.Renderer(model,height=224,width=224)
fisheye=None
if args.camera_profile=='approximate-fisheye':
    from fisheye_camera import FisheyeCamera
    fisheye=FisheyeCamera(model,'head_camera')
view=mujoco.MjvCamera();ctx=zmq.Context();head,wrist,hand=[ctx.socket(zmq.PUB) for _ in range(3)]
for socket,port in [(head,20555),(wrist,20554),(hand,20558)]:socket.bind(f'tcp://127.0.0.1:{port}')
video=cv2.VideoWriter(str(run/'simulation.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(448,448))
labels={'vla':'OFFICIAL VLA / SIM','baseline':'POSE BASELINE','replay':'RECORDED ACTION REPLAY','calibration':'PRIVILEGED TCP / NOT VLA','tracking-probe':'JOINT TRACKING PROBE','scripted-baseline':'G1 SCRIPT BASELINE / NOT VLA'}
spec=json.loads((run/'task_spec.json').read_text()) if (run/'task_spec.json').exists() else {}
if spec.get('protocol')=='paper_task_proxy_v1':
    labels['vla']='VLA / PAPER SCENE PROXY'
    labels['baseline']='PAPER SCENE / STAND TEST'
if os.environ.get('OPENHLM_ARM_REFERENCE_SERVO')=='1':
    labels['scripted-baseline']='G1 SCRIPT / ARM SERVO / NOT VLA'
    labels['vla']='OFFICIAL VLA / ARM SERVO / SIM'
stamps=[];video_frames=0;next_video=0.
try:
    while not (run/'render-stop').exists():
        tick=time.monotonic()
        try:s=json.loads((run/'status.json').read_text())
        except FileNotFoundError:time.sleep(.01);continue
        data.qpos[:]=s['qpos'];data.qvel[:]=s['qvel'];data.time=s['sim_time'];mujoco.mj_forward(model,data)
        task=s.get('task',{});images=[]
        for name,r in [('head_camera',head_renderer),('eval_left_wrist',wrist_renderer),('eval_right_wrist',wrist_renderer)]:
            if name=='head_camera' and fisheye:images.append(fisheye.render(data))
            else:r.update_scene(data,camera=name);images.append(r.render().copy())
        bgr=[cv2.cvtColor(im,cv2.COLOR_RGB2BGR) for im in images]
        head.send(struct.pack('iii',464,200,3)+np.concatenate([bgr[0],bgr[0]],axis=1).tobytes())
        ok,jpeg=cv2.imencode('.jpg',np.concatenate(bgr[1:],axis=1))
        if ok:wrist.send(struct.pack('iii',640,240,len(jpeg))+jpeg.tobytes())
        opening=task.get('gripper_openings',{})
        hand.send(msgpack.packb({'state_hand_left':[opening.get('lh',1.)],'state_hand_right':[opening.get('rh',1.)]}))
        if s['released'] and tick>=next_video:
            view.lookat[:]=[.35,-.20,.65] if task else data.qpos[:3]
            view.distance,view.azimuth,view.elevation=(2.6,140,-20) if task else (2.5,135,-15)
            renderer.update_scene(data,camera=view);external=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
            if next_video==0:next_video=tick
            previews=[]
            for im in bgr:
                h,w=im.shape[:2];scale=224/max(h,w);small=cv2.resize(im,(int(w*scale),int(h*scale)));panel=np.zeros((224,224,3),np.uint8)
                y,x=(224-small.shape[0])//2,(224-small.shape[1])//2;panel[y:y+small.shape[0],x:x+small.shape[1]]=small;previews.append(panel)
            for panel,label in zip([external,*previews],[labels[args.run_label] if s['evaluating'] else 'INITIALIZATION / NOT VLA','HEAD (APPROX FISHEYE)','LEFT WRIST','RIGHT WRIST']):
                cv2.putText(panel,label,(5,15),cv2.FONT_HERSHEY_SIMPLEX,.34,(255,255,255),1,cv2.LINE_AA)
            frame=np.concatenate([np.concatenate([external,previews[0]],1),np.concatenate(previews[1:],1)],0)
            if task:
                label=(f"PAPER PROXY {task['paper_points_total']}/4 | FOOT {int(task['left_foot_pedal_open'])} HOME {int(task['returned_home'])}"
                       if 'paper_points_total' in task else f"GRASP {int(task['grasped'])} LIFT {int(task['lifted'])} PLACE {int(task['success'])}")
                cv2.putText(frame,label,(5,441),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
            while tick>=next_video:video.write(frame);video_frames+=1;next_video+=.1
        stamps.append(tick)
        time.sleep(max(0,1/30-(time.monotonic()-tick)))
finally:
    (run/'render_report.json').write_text(json.dumps({'frames':len(stamps),'video_frames':video_frames,'camera_actual_hz':(len(stamps)-1)/(stamps[-1]-stamps[0]) if len(stamps)>1 else 0},indent=2))
    video.release();renderer.close();head_renderer.close();wrist_renderer.close()
    if fisheye:fisheye.close()
    for socket in [head,wrist,hand]:socket.close(linger=0)
    ctx.term()
