"""Positive control: scripted gripper carriage, physical bottle, no object weld.

This tests scene/gripper/scoring mechanics. It is not a G1 or VLA success trial.
"""
import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from bottle_task import BottleTaskPhysics, SPEC, vector

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--video", action="store_true")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
scene = ET.parse(args.scene)
world_source = scene.getroot().find("worldbody")
root = ET.Element("mujoco", model="gripper_mechanics_positive_control")
ET.SubElement(root, "option", timestep=".002", integrator="implicitfast", iterations="80")
visual = ET.SubElement(root, "visual")
ET.SubElement(visual, "headlight", ambient=".45 .45 .45", diffuse=".8 .8 .8", specular=".1 .1 .1")
world = ET.SubElement(root, "worldbody")
ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse=".8 .8 .8", ambient=".3 .3 .3")
ET.SubElement(world, "light", pos="1 1 2", dir="-1 -1 -1", diffuse=".6 .6 .6", castshadow="false")
ET.SubElement(world, "geom", type="plane", size="2 2 .1", rgba=".25 .3 .35 1", contype="8", conaffinity="15")
for name in ["task_platform", "task_bin", "task_bottle", "rh_tool"]:
    body = copy.deepcopy(world_source.find(f"body[@name='{name}']"))
    world.append(body)
tool = world.find("body[@name='rh_tool']")
bottle = np.array(SPEC["bottle_initial"], dtype=float)
grasp = bottle - np.array([.065, 0, 0])
start = grasp - np.array([.12, 0, 0])
tool.set("pos", vector(start)); tool.set("quat", "1 0 0 0")
ET.SubElement(tool, "site", name="rh_tcp", pos=".065 0 0", size=".003")
ET.SubElement(world, "body", name="carriage", mocap="true", pos=vector(start))
eq = ET.SubElement(root, "equality")
ET.SubElement(eq, "weld", body1="carriage", body2="rh_tool", relpose="0 0 0 1 0 0 0", solref=".01 1")
path = args.output_dir / "mechanics.xml"
ET.ElementTree(root).write(path)
model = mujoco.MjModel.from_xml_path(str(path)); data = mujoco.MjData(model)
physics = BottleTaskPhysics(model, data, sides=("rh",))
lift = grasp.copy(); lift[2] = 1.08
over_bin = lift.copy(); over_bin[:2] = [SPEC["bin_center"][0] - .065, SPEC["bin_center"][1]]
release = over_bin.copy(); release[2] = .90
withdraw = release.copy(); withdraw[2] = 1.18
waypoints = [(0., start, 1.), (.5, start, 1.), (2.5, grasp, 1.), (4., grasp, 0.),
             (6., lift, 0.), (9., over_bin, 0.), (10., release, 0.),
             (11., release, 1.), (12., withdraw, 1.), (15., withdraw, 1.)]
renderer = writer = None
if args.video:
    import cv2
    renderer = mujoco.Renderer(model, height=480, width=640)
    writer = cv2.VideoWriter(str(args.output_dir / "mechanics.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480))
    camera = mujoco.MjvCamera(); camera.lookat[:] = [.48, -.2, .65]
    camera.distance, camera.azimuth, camera.elevation = 1.8, 140, -65
records, next_frame = [], 0.
while data.time < 15.:
    t = float(data.time)
    for (ta, pa, ca), (tb, pb, cb) in zip(waypoints, waypoints[1:]):
        if ta <= t <= tb:
            a = np.clip((t - ta) / (tb - ta), 0, 1); a = a*a*(3 - 2*a)
            data.mocap_pos[0] = (1-a)*pa + a*pb
            physics.command["rh"] = float((1-a)*ca+a*cb)
            break
    physics.before_step(); mujoco.mj_step(model, data)
    measurement = physics.measure()
    if data.time >= next_frame:
        records.append({"time": float(data.time), **measurement})
        if renderer:
            renderer.update_scene(data, camera=camera)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (0, 0), (640, 42), (25, 25, 25), -1)
            cv2.putText(frame, "SCRIPTED GRIPPER PHYSICS - NOT VLA / G1", (10, 17), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
            label = f"t={data.time:.1f}s grasp={physics.score.grasped} lift={physics.score.lifted} placed={physics.score.success}"
            cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
            writer.write(frame)
        next_frame += .05
result = {"test": "scripted_gripper_physics_positive_control", "not_a_vla_or_g1_trial": True,
          "bottle_is_free_body": True, "object_welds": 0, **physics.score.result()}
(args.output_dir / "mechanics.json").write_text(json.dumps(result, indent=2))
(args.output_dir / "telemetry.json").write_text(json.dumps(records))
print(json.dumps(result, indent=2))
if renderer:
    renderer.close(); writer.release()
raise SystemExit(0 if result["success"] else 1)
