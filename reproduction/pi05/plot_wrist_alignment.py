"""Scientific projection diagnostic; annotated references are not policy inputs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
base=Path(__file__).resolve().parent
out=base/'results/tcp_alignment';fit=json.loads((base/'wrist_camera_visual_fit.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(12,5.5))
for ax,side in zip(axes,['left','right']):
    c=fit['cameras'][side];im=Image.open(out/'reference'/f'{side}_0.jpg')
    ax.imshow(im);ax.set_xlim(0,640);ax.set_ylim(560,0)
    keep=c['fit_landmark_indices'];pixels=np.array(c['observed_landmarks'])[keep];pred=np.array(c['projected_landmarks'])[keep]
    ax.scatter(pixels[:,0],pixels[:,1],s=100,facecolors='none',edgecolors='lime',label='Public video labels')
    ax.scatter(pred[:,0],pred[:,1],s=28,color='blue',marker='+',label='Fitted surrogate')
    ax.scatter([138.09,501.91],[54.44,54.44],color='red',marker='x',label='Previous inner tips')
    ax.scatter([320],[251.83],color='orange',marker='*',s=100,label='Previous TCP')
    ax.scatter(*c['projected_tcp_640x480'],color='magenta',marker='*',s=100,label='Fitted TCP')
    ax.axhline(480,color='gray',ls='--',lw=1);ax.set_title(f'{side} wrist: tip fit RMS {c["landmark_rms_px"]:.1f}px')
    ax.set_xlabel('u / pixels');ax.set_ylabel('v / pixels')
axes[0].legend(loc='upper right',fontsize=8)
fig.suptitle('Surrogate camera-to-jaw projection check | not factory hand-eye calibration',fontsize=12)
fig.tight_layout();fig.savefig(out/'camera_projection_check.png',dpi=160)
