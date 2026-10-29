"""Approximate equidistant 180 degree lens from five perspective cube faces.

This matches the projection family visible in the public head videos. It is NOT
hardware calibration: lens coefficients, optical center and mount are unknown.
"""
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class FisheyeCamera:
    def __init__(self, model, camera_name, height=200, width=232, face_size=192):
        self.model, self.id = model, model.camera(camera_name).id
        self.original_quat = model.cam_quat[self.id].copy()
        self.original_fovy = model.cam_fovy[self.id]
        self.renderer = mujoco.Renderer(model, height=face_size, width=face_size)
        self.faces = [np.eye(3),
                      np.array([[0,0,-1],[0,1,0],[1,0,0]]),
                      np.array([[0,0,1],[0,1,0],[-1,0,0]]),
                      np.array([[1,0,0],[0,0,-1],[0,1,0]]),
                      np.array([[1,0,0],[0,0,1],[0,-1,0]])]
        self.base = Rotation.from_quat(self.original_quat[[1,2,3,0]]).as_matrix()
        yy,xx=np.mgrid[:height,:width]
        x=(xx-(width-1)/2)/(width/2); y=(-(yy-(height-1)/2))/(height/2)
        r=np.sqrt(x*x+y*y);theta=np.minimum(r,1)*np.pi/2
        ray=np.stack([np.sin(theta)*x/np.maximum(r,1e-8),np.sin(theta)*y/np.maximum(r,1e-8),-np.cos(theta)],-1)
        self.shape=(height,width,3);self.maps=[]
        depths=np.stack([-(ray @ face)[...,2] for face in self.faces],-1)
        selected=np.argmax(depths,-1)
        for i,face in enumerate(self.faces):
            local=ray @ face;depth=np.maximum(-local[...,2],1e-6)
            u=((local[...,0]/depth+1)*face_size/2-.5).astype(np.float32)
            v=((1-local[...,1]/depth)*face_size/2-.5).astype(np.float32)
            self.maps.append((u,v,(selected==i)&(r<=1)))

    def render(self, data):
        result=np.zeros(self.shape,dtype=np.uint8)
        self.model.cam_fovy[self.id]=90
        try:
            # Geometry is unchanged between cube faces. Update it once, then
            # rotate the two GL cameras without rerunning robot/contact physics.
            self.renderer.update_scene(data,camera=self.id)
            base_world=data.cam_xmat[self.id].reshape(3,3).copy()
            eye=data.cam_xpos[self.id].copy()
            for face,(u,v,mask) in zip(self.faces,self.maps):
                world=base_world @ face
                for camera in self.renderer.scene.camera:
                    camera.pos[:]=eye
                    camera.forward[:]=-world[:,2]
                    camera.up[:]=world[:,1]
                frame=cv2.remap(self.renderer.render(),u,v,cv2.INTER_LINEAR)
                result[mask]=frame[mask]
        finally:
            self.model.cam_fovy[self.id]=self.original_fovy
        return result

    def close(self):
        self.renderer.close()
