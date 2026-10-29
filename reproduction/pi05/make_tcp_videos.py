"""Annotate recorded physics video with measured TCP/contact data (not inputs)."""
import csv,json,subprocess
from pathlib import Path
import cv2,numpy as np
root=Path(__file__).resolve().parents[2]
cases=[('g1_tcp_fov58_vla01','official_vla_excerpt.mp4',False),
       ('g1_tcp_calibration02','contact_diagnostic_not_vla.mp4',True)]
out=root/'logs/tcp_alignment_delivery';out.mkdir(exist_ok=True)
metadata=[]
for name,filename,calibration in cases:
    p=root/'logs'/name
    rows=json.loads((p/'task_telemetry.json').read_text());ev=[x for x in rows if x['evaluating']]
    telemetry=list(csv.DictReader((p/'telemetry.csv').open()))
    origin=next(float(x['wall_time']) for x in telemetry if x['released']=='True')
    contacts=[x for x in ev if max(x['tool_contact_forces'].values())>.1]
    closing=[x for x in ev if x['gripper_commands']['rh']<.5]
    focus=contacts[0] if calibration and contacts else min(closing or ev,key=lambda x:np.linalg.norm(x['tcp_diagnostics']['rh']['bottle_in_tcp_frame']))
    start=max(0,focus['wall_time']-origin-5);end=start+25
    cap=cv2.VideoCapture(str(p/'simulation.mp4'));fps=cap.get(cv2.CAP_PROP_FPS)
    writer=cv2.VideoWriter(str(out/'annotated_temp.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(448,448))
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(start*fps));index=int(start*fps)
    stamps=np.array([x['wall_time'] for x in rows])
    while index/fps<end:
        ok,frame=cap.read()
        if not ok:break
        wall=origin+index/fps;r=rows[int(np.argmin(abs(stamps-wall)))];dist=np.linalg.norm(r['tcp_diagnostics']['rh']['bottle_in_tcp_frame'])
        cv2.rectangle(frame,(0,398),(448,432),(15,15,15),-1)
        label='TCP CALIBRATION - NOT VLA' if calibration else 'OFFICIAL VLA - UNSUCCESSFUL ATTEMPT'
        cv2.putText(frame,label,(5,411),cv2.FONT_HERSHEY_SIMPLEX,.37,(255,255,255),1,cv2.LINE_AA)
        text=f"TCP-bottle {dist:.3f}m  grip {r['gripper_commands']['rh']:.2f}  fingers {r['finger_contact_forces']['rh'][0]:.1f}/{r['finger_contact_forces']['rh'][1]:.1f}N"
        cv2.putText(frame,text,(5,426),cv2.FONT_HERSHEY_SIMPLEX,.34,(0,255,255),1,cv2.LINE_AA)
        writer.write(frame);index+=1
    cap.release();writer.release()
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(out/'annotated_temp.mp4'),'-c:v','libx264','-preset','fast','-crf','22','-movflags','+faststart',str(out/filename)],check=True)
    metadata.append({'source_run':name,'output':filename,'source_video_start_s':start,'duration_s':index/fps-start,
                     'calibration_not_vla':calibration,'focus_time_after_evaluation_s':focus['wall_time']-ev[0]['wall_time']})
(out/'annotated_temp.mp4').unlink();(out/'video_metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2))
