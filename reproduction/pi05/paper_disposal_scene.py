"""Bottle Disposal task topology from OpenHLM Appendix A (p.18).

Metric dimensions and the five layouts are explicit proxy assumptions, not
released author CAD/calibration or the paper's unpublished five scene photos.
"""
import json, math, copy
import xml.etree.ElementTree as ET
from bottle_task import build_task, vector

from paper_protocol import PAPER_PROMPT
LAYOUTS=[
    {'bottle':[.70,-.38], 'can':[.96,-.38], 'mug':[.85,-.67]},
    {'bottle':[.76,-.46], 'can':[.62,-.66], 'mug':[.98,-.35]},
    {'bottle':[.66,-.54], 'can':[.99,-.63], 'mug':[.88,-.34]},
    {'bottle':[.81,-.39], 'can':[.64,-.62], 'mug':[1.00,-.65]},
    {'bottle':[.73,-.60], 'can':[.97,-.36], 'mug':[.62,-.38]},
]

def add_pedal_bin(world,equality,center=(.80,.08),height=.66,radius=.18,contact=None):
    """Passive joint linkage: only physical pedal displacement opens the lid."""
    body=ET.SubElement(world,'body',name='task_bin',pos=vector([*center,0]))
    ET.SubElement(body,'geom',name='bin_base',type='cylinder',pos='0 0 .02',size=f'{radius+.02} .02',
                  rgba='.58 .44 .18 1',contype='8',conaffinity='15')
    for i in range(16):
        angle=2*math.pi*i/16;r=radius+.01
        ET.SubElement(body,'geom',name=f'bin_wall_{i}',type='box',
            pos=vector([r*math.cos(angle),r*math.sin(angle),(height+.04)/2]),
            size=vector([.010,r*math.tan(math.pi/16)+.001,(height-.04)/2]),
            euler=f'0 0 {angle}',rgba='.64 .49 .20 1',contype='8',conaffinity='15',friction='.9 .005 .0005')
    hinge_x=radius+.025
    lid=ET.SubElement(body,'body',name='bin_lid',pos=vector([hinge_x,0,height+.01]))
    ET.SubElement(lid,'joint',name='bin_lid_hinge',type='hinge',axis='0 1 0',range='0 1.45',
                  damping='.15',stiffness='.08',springref='0',armature='.002')
    ET.SubElement(lid,'geom',name='bin_lid_plate',type='cylinder',pos=f'{-hinge_x} 0 0',
                  size=f'{radius+.022} .009',mass='.22',rgba='.09 .09 .09 1',contype='8',conaffinity='15')
    pedal=ET.SubElement(body,'body',name='bin_pedal',pos=vector([-radius+.015,0,.060]))
    ET.SubElement(pedal,'joint',name='bin_pedal_hinge',type='hinge',axis='0 -1 0',range='0 .35',
                  damping='.10',stiffness='1',springref='0',armature='.002')
    ET.SubElement(pedal,'geom',name='bin_pedal_pad',type='box',pos='-.07 0 0',size='.060 .070 .010',mass='.10',
                  rgba='.08 .08 .08 1',contype='8',conaffinity='15',friction='1 .005 .0005')
    # A passive joint constraint represents the internal lever, not an actuator.
    ET.SubElement(equality,'joint',name='bin_pedal_linkage',joint1='bin_lid_hinge',joint2='bin_pedal_hinge',
                  polycoef='0 4 0 0 0',solref='.008 1')
    if contact is not None:
        # The pedal passes through its unmodelled rear bearing/housing. Avoid
        # jamming the lever against the coarse bin-wall proxy, while retaining
        # all shoe/pedal and bottle/lid contacts.
        ET.SubElement(contact,'exclude',body1='task_bin',body2='bin_pedal')
    return body

def build_paper_task(robot,scene,run,tool_profile='visual-fit',layout_id=0):
    if not 0<=layout_id<len(LAYOUTS):raise ValueError('Paper proxy has five layouts, numbered 0..4')
    build_task(robot,scene,run,tool_profile=tool_profile)
    root=scene.getroot();world=root.find('worldbody');layout=LAYOUTS[layout_id]
    table=world.find("body[@name='task_platform']");table.set('pos','.83 -.52 0')
    for geom in table.findall('geom'):geom.set('rgba','.88 .88 .87 1')
    world.remove(world.find("body[@name='task_bin']"))
    contact=root.find('contact')
    if contact is None:contact=ET.SubElement(root,'contact')
    add_pedal_bin(world,root.find('equality'),contact=contact)
    bottle=world.find("body[@name='task_bottle']");bottle.set('pos',vector([*layout['bottle'],.842]))
    for kind,xy,color,size in [('can',layout['can'],'.75 .03 .02 1','.032 .065'),
                                ('mug',layout['mug'],'.85 .72 .08 1','.042 .045')]:
        body=ET.SubElement(world,'body',name=f'distractor_{kind}',pos=vector([*xy,.74+float(size.split()[1])+.002]))
        ET.SubElement(body,'freejoint',name=f'distractor_{kind}_free')
        ET.SubElement(body,'geom',name=f'distractor_{kind}_geom',type='cylinder',size=size,mass='.12',
                      rgba=color,contype='4',conaffinity='15',friction='1 .005 .0005')
    # A coarse lab backdrop follows the task photographs; no photometric match claimed.
    sofa=ET.SubElement(world,'body',name='proxy_sofa',pos='1.65 -.15 0')
    ET.SubElement(sofa,'geom',type='box',pos='0 0 .25',size='.28 1.10 .25',rgba='.28 .29 .30 1')
    ET.SubElement(sofa,'geom',type='box',pos='.22 0 .64',size='.06 1.10 .16',rgba='.82 .69 .04 1')
    texture=root.find(".//texture[@name='groundplane']")
    if texture is not None:
        texture.set('rgb1','.65 .66 .67');texture.set('rgb2','.69 .70 .71')
    spec=json.loads((run/'task_spec.json').read_text())
    spec.update({'task_id':'hlm12_bottle_disposal','protocol':'paper_task_proxy_v1','layout_id':layout_id,
        'paper_source':'https://arxiv.org/pdf/2606.22174v1#page=18', 'paper_prompt':PAPER_PROMPT,
        'bottle_initial':[*layout['bottle'],.842], 'bin_center':[.80,.08],'bin_half_size':[.18,.18],
        'bin_radius':.18,'bin_rim':.66,'bin_floor':.04,'home_xy':[-.65,0],
        'pedal_nominal_center_z':.06,
        'approach_xy':[.25,0],'home_tolerance_m':.18,'approach_tolerance_m':.20,
        'metric_dimensions':'provisional proxy assumptions; not author measurements',
        'layout_provenance':'five deterministic proxy layouts; author evaluation photos unavailable',
        'camera_calibration':'approximate; hardware intrinsics and hand-eye transforms unavailable',
        'rubric':['approach_and_return','green_bottle_grasp','left_foot_pedal_open','bottle_inside_bin'],
        'strict_paper_replication':False})
    (run/'task_spec.json').write_text(json.dumps(spec,indent=2))
    return spec
