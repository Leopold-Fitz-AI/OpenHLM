"""Passive bin unit fixture. This is NOT a G1 or VLA task rollout."""
import argparse,json,math
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco,numpy as np
from paper_disposal_scene import add_pedal_bin
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
r=ET.Element('mujoco',model='paper_bin_mechanics_fixture');ET.SubElement(r,'compiler',angle='radian')
ET.SubElement(r,'option',timestep='.002',integrator='implicitfast',cone='elliptic',impratio='10',iterations='80')
w=ET.SubElement(r,'worldbody');ET.SubElement(w,'light',pos='0 -1 3');ET.SubElement(w,'geom',type='plane',size='2 2 .1')
eq=ET.SubElement(r,'equality');add_pedal_bin(w,eq,contact=ET.SubElement(r,'contact'))
foot=ET.SubElement(w,'body',name='fixture_foot',mocap='true',pos='.535 .08 .14')
ET.SubElement(foot,'geom',name='fixture_sole',type='box',size='.06 .045 .015',contype='1',conaffinity='8')
b=ET.SubElement(w,'body',name='probe_bottle',pos='.80 .08 .90');ET.SubElement(b,'freejoint',name='probe_free')
ET.SubElement(b,'geom',type='cylinder',size='.03 .10',mass='.12',contype='4',conaffinity='8',rgba='0 .6 .1 1')
path=a.output/'fixture.xml';ET.ElementTree(r).write(path)
m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
lid=m.jnt_qposadr[m.joint('bin_lid_hinge').id];pedal=m.jnt_qposadr[m.joint('bin_pedal_hinge').id];bq=m.jnt_qposadr[m.joint('probe_free').id]
# Closed lid must block a dropped free bottle.
for _ in range(750):mujoco.mj_step(m,d)
closed_bottom=float(d.qpos[bq+2]-.1);assert closed_bottom>.64,closed_bottom
# Independent press/release fixture without the bottle resting on the lid.
d=mujoco.MjData(m);d.qpos[bq:bq+3]=[1.3,.08,.102]
trace=[]
for _ in range(3000):
 t=d.time;f=np.clip((t-.5),0,1) if t<3 else max(0,1-(t-3))
 d.mocap_pos[0]=[.535,.08,.14-.083*f];mujoco.mj_step(m,d)
 trace.append([float(d.time),float(d.qpos[lid]),float(d.qpos[pedal])])
np.savez(a.output/'final_state.npz',qpos=d.qpos,qvel=d.qvel,mocap=d.mocap_pos,passive=d.qfrc_passive,constraint=d.qfrc_constraint);trace=np.array(trace);np.save(a.output/'pedal_trace.npy',trace);maximum=float(trace[:,1].max());end=float(trace[-1,1]);assert maximum>1.,maximum;assert end<.08,end
# A consistent open-lid initial condition lets the same free bottle fall inside.
d=mujoco.MjData(m);d.qpos[lid]=1.36;d.qpos[pedal]=.34;d.mocap_pos[0]=[.535,.08,.057];mujoco.mj_forward(m,d)
for _ in range(1250):mujoco.mj_step(m,d)
opened_bottom=float(d.qpos[bq+2]-.1);assert .02<opened_bottom<.07,opened_bottom
out={'test':'passive_pedal_bin_unit_fixture','not_robot_or_vla_success':True,'lid_max_angle_rad':maximum,'lid_angle_after_release_rad':end,'closed_lid_bottle_bottom_m':closed_bottom,'open_lid_bottle_bottom_m':opened_bottom,'actuators':m.nu,'tests_passed':True}
(a.output/'mechanics.json').write_text(json.dumps(out,indent=2));np.save(a.output/'pedal_trace.npy',trace);print(json.dumps(out,indent=2))
