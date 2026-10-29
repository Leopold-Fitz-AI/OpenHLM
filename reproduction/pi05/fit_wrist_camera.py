"""Fit surrogate camera-to-jaw projection to manually labelled public video tips.

This is a visual surrogate fit, not a recovered metric hardware calibration.
Only rigid jaw landmarks are used; no bottle/task outcome participates.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

parser=argparse.ArgumentParser()
parser.add_argument('--fovy-prior',type=float,default=75)
parser.add_argument('--output',default='wrist_camera_visual_fit.json')
args=parser.parse_args()

# Distal +/-Y jaw cross-sections in the existing surrogate wrist frame.
points=[]
for sign in [1,-1]:
    for y,z in [(sign*.048,.027),(sign*.055,.027),(sign*.048,-.027),(sign*.055,-.027)]:
        points.append([.170,y,z])
points=np.asarray(points)
labels={
    'right':[[107,311],[87,302],[109,366],[85,354], [463,302],[492,294],[463,351],[499,341]],
    'left': [[176,265],[151,262],[182,334],[155,328], [524,256],[550,254],[518,327],[545,322]]}
def camera(params):
    x,y,z,theta,yaw,roll,fovy=params
    # columns: screen right, screen up, backward
    base=np.array([[0,np.sin(theta),-np.cos(theta)],[-1,0,0],[0,np.cos(theta),np.sin(theta)]])
    R=Rotation.from_euler('z',yaw).as_matrix() @ base @ Rotation.from_euler('z',roll).as_matrix()
    return np.array([x,y,z]),R,240/np.tan(np.deg2rad(fovy)/2)
def project(params,p):
    pos,R,f=camera(params);v=(p-pos)@R;depth=-v[:,2]
    return np.stack([320+f*v[:,0]/depth,240-f*v[:,1]/depth],1),depth
out={}
for side,pixels in labels.items():
    pixels=np.asarray(pixels,dtype=float)
    # Only distal top corners are unambiguous in the video; lower edges are
    # occluded/rounded and are retained solely for visual review.
    keep=np.array([0,1,4,5]); fit_pixels=pixels[keep]; fit_points=points[keep]
    def residual(a):
        uv,depth=project(a,fit_points)
        # Weak mount/FOV priors resolve scale/rotation ambiguity; landmarks
        # carry +/-~10px annotation uncertainty and approximate box geometry.
        # Downward lens prior also follows the all-floor initial public view;
        # four coplanar tips alone do not determine extrinsics uniquely.
        prior=np.array([(a[0]-.15)*20,a[1]*10,(a[2]-.12)*20,(a[6]-args.fovy_prior)*2,(a[3]-1.1)*100])
        return np.r_[(uv-fit_pixels).ravel(),prior,np.maximum(.02-depth,0)*10000]
    initial=[.15,0,.12,1.1,0,0,args.fovy_prior]
    sol=least_squares(residual,initial,bounds=([-.35,-.12,.03,.02,-.6,-.5,45],[.30,.12,.35,1.5,.6,.5,115]),max_nfev=3000)
    pos,R,f=camera(sol.x);uv,_=project(sol.x,points);tcp,_=project(sol.x,np.array([[.125,0,0]]))
    out[side]={'position':pos.tolist(),'xyaxes':np.r_[R[:,0],R[:,1]].tolist(),
               'fovy':float(sol.x[6]),'landmark_rms_px':float(np.sqrt(np.mean((uv[keep]-fit_pixels)**2))),
               'projected_tcp_640x480':tcp[0].tolist(),'observed_landmarks':pixels.tolist(),'fit_landmark_indices':keep.tolist(),
               'projected_landmarks':uv.tolist(),'parameters':sol.x.tolist()}
p=Path(__file__).with_name(args.output);p.write_text(json.dumps({'method':'approximate_manual_jaw_landmark_fit_not_hardware_calibration','cameras':out},indent=2));print(p);print(json.dumps(out,indent=2))
