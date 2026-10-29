"""Physical bottle/bin scene and a parallel-jaw surrogate for CTAG grippers.

Only the tool mounts are welded to the wrists. The bottle is always a free body.
Extra tool joints follow the original robot joints, preserving SDK index mapping.
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import zmq
from task_metrics import BottleBinScore, point_aabb_distance

SPEC = {"table_top": .74, "bottle_initial": [.47, -.12, .842],
        "bin_center": [.47, -.50], "bin_half_size": [.15, .15],
        "bin_floor": .04, "bin_rim": .62, "max_jaw_travel": .045}


def vector(values):
    return " ".join(map(str, values))


def build_task(robot, scene, run: Path, tool_profile="legacy"):
    pad_half_width = .0035 if tool_profile == "visual-fit" else .006
    jaw_base_offset = pad_half_width + .003
    for body in robot.findall(".//body"):
        if "hand" in body.get("name", ""):
            for geom in body.findall(".//geom"):
                geom.set("contype", "0"); geom.set("conaffinity", "0")
                geom.set("rgba", "0.6 0.6 0.6 0")
    for side, short in [("left", "lh"), ("right", "rh")]:
        wrist = robot.find(f".//body[@name='{side}_wrist_yaw_link']")
        for geom in wrist.findall("geom"):
            if "hand" in geom.get("mesh", ""):
                geom.set("contype", "0"); geom.set("conaffinity", "0")
                geom.set("rgba", "0.6 0.6 0.6 0")
        ET.SubElement(wrist, "site", name=f"{short}_tcp", pos="0.125 0 0", size="0.003", rgba="1 0.8 0 0")
    robot.find(".//camera[@name='head_camera']").set("fovy", "100")
    robot.write(run / "robot.xml")
    original = mujoco.MjModel.from_xml_path(str(run / "robot.xml"))
    initial = mujoco.MjData(original); mujoco.mj_forward(original, initial)
    world = scene.getroot().find("worldbody")
    world.find("geom[@name='floor']").set("contype", "8")
    world.find("geom[@name='floor']").set("conaffinity", "15")
    equality = ET.SubElement(scene.getroot(), "equality")
    for side, short in [("left", "lh"), ("right", "rh")]:
        wrist_id = original.body(f"{side}_wrist_yaw_link").id
        position = initial.xpos[wrist_id] + initial.xmat[wrist_id].reshape(3, 3) @ np.array([.06, 0, 0])
        body = ET.SubElement(world, "body", name=f"{short}_tool", pos=vector(position), quat=vector(initial.xquat[wrist_id]))
        ET.SubElement(body, "freejoint", name=f"{short}_tool_free")
        ET.SubElement(body, "geom", name=f"{short}_tool_base", type="box", size=".025 .065 .025", pos="-.005 0 0",
                      mass=".08", rgba=".15 .2 .26 1", contype="2", conaffinity="12")
        for label, sign in [("a", 1), ("b", -1)]:
            jaw = ET.SubElement(body, "body", name=f"{short}_jaw_{label}", pos=f".02 {sign * jaw_base_offset} 0")
            ET.SubElement(jaw, "joint", name=f"{short}_jaw_{label}_slide", type="slide", axis=f"0 {sign} 0",
                          range="0 .045", limited="true", damping="1.5", armature=".003")
            ET.SubElement(jaw, "geom", name=f"{short}_pad_{label}", type="box", pos=".045 0 0", size=f".045 {pad_half_width} .027",
                          mass=".035", rgba=".08 .09 .10 1", friction="1.3 .005 .0005", condim="4",
                          contype="2", conaffinity="12", solref=".012 1")
        ET.SubElement(equality, "weld", name=f"{short}_tool_mount", body1=f"{side}_wrist_yaw_link", body2=f"{short}_tool",
                      relpose=".06 0 0 1 0 0 0", solref=".01 1")
    table = ET.SubElement(world, "body", name="task_platform", pos=".60 0 0")
    ET.SubElement(table, "geom", name="task_tabletop", type="box", pos="0 0 .715", size=".27 .30 .025",
                  rgba=".45 .33 .20 1", friction=".8 .005 .0005", contype="8", conaffinity="15")
    for x in [-.22, .22]:
        for y in [-.25, .25]:
            ET.SubElement(table, "geom", type="box", pos=f"{x} {y} .345", size=".025 .025 .345",
                          rgba=".23 .24 .25 1", contype="8", conaffinity="15")
    bin_body = ET.SubElement(world, "body", name="task_bin", pos=vector([*SPEC["bin_center"], 0]))
    ET.SubElement(bin_body, "geom", name="bin_base", type="box", pos="0 0 .02", size=".174 .174 .02",
                  rgba=".18 .25 .32 1", contype="8", conaffinity="15")
    for axis in [0, 1]:
        for sign in [-1, 1]:
            pos = [0, 0, .33]; pos[axis] = sign * .162
            size = [.174, .174, .29]; size[axis] = .012
            ET.SubElement(bin_body, "geom", name=f"bin_wall_{axis}_{sign}", type="box", pos=vector(pos), size=vector(size),
                          rgba=".22 .32 .40 1", contype="8", conaffinity="15", friction=".9 .005 .0005")
    bottle = ET.SubElement(world, "body", name="task_bottle", pos=vector(SPEC["bottle_initial"]))
    ET.SubElement(bottle, "freejoint", name="bottle_free")
    for name, pos, size, mass, rgba in [
        ("bottle_body", "0 0 0", ".03 .10", ".10", ".03 .55 .12 1"),
        ("bottle_neck", "0 0 .118", ".014 .018", ".015", ".03 .48 .1 1"),
        ("bottle_cap", "0 0 .142", ".017 .006", ".005", ".08 .12 .15 1"),
    ]:
        ET.SubElement(bottle, "geom", name=name, type="cylinder", pos=pos, size=size, mass=mass, rgba=rgba,
                      friction="1.1 .005 .0005", condim="4", contype="4", conaffinity="11", solref=".012 1")
    scene.getroot().find("include").set("file", str(run / "robot.xml"))
    (run / "task_spec.json").write_text(json.dumps({**SPEC, "tool_profile": tool_profile, "pad_half_width": pad_half_width, "tcp_in_wrist": [.125,0,0]}, indent=2))


class BottleTaskPhysics:
    def __init__(self, model, data, context=None, sides=("lh", "rh"),spec=None):
        self.model, self.data = model, data
        self.spec=spec or SPEC
        self.command = {side: 1. for side in sides}
        self.sub = context.socket(zmq.SUB) if context else None
        if self.sub is not None:
            self.sub.setsockopt(zmq.SUBSCRIBE, b"pose")
            self.sub.setsockopt(zmq.CONFLATE, 1)
            self.sub.connect("tcp://127.0.0.1:20556")
        self.jaws = {s: [model.joint(f"{s}_jaw_{a}_slide").id for a in ("a", "b")] for s in self.command}
        self.pads = {s: [model.geom(f"{s}_pad_{a}").id for a in ("a", "b")] for s in self.command}
        self.tool_geoms = {s: self.pads[s] + [model.geom(f"{s}_tool_base").id] for s in self.command}
        self.bottle_geoms = [model.geom(n).id for n in ("bottle_body", "bottle_neck", "bottle_cap")]
        self.bottle_body = model.body("task_bottle").id
        self.bottle_joint = model.joint("bottle_free").id
        self.mechanics = BottleBinScore(self.spec["table_top"], tuple(self.spec["bin_center"]), tuple(self.spec["bin_half_size"]),
                                    self.spec["bin_floor"], self.spec["bin_rim"],bin_radius=self.spec.get("bin_radius"))
        self.score=self.mechanics
        self.paper_task=self.spec.get('protocol')=='paper_task_proxy_v1'
        if self.paper_task:
            from paper_task_metrics import PaperDisposalScore
            self.score=PaperDisposalScore(self.mechanics,tuple(self.spec['home_xy']),tuple(self.spec['approach_xy']),
                self.spec['home_tolerance_m'],self.spec['approach_tolerance_m'])
            self.left_foot_geoms={g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.startswith('left_ankle')}
            self.pedal_geom=model.geom('bin_pedal_pad').id
            self.lid_q=model.jnt_qposadr[model.joint('bin_lid_hinge').id]
        self.active_side = None
        self.last = {}
        self.contact_events = []
        for ids in self.jaws.values():
            for joint in ids:
                data.qpos[model.jnt_qposadr[joint]] = SPEC["max_jaw_travel"]
        mujoco.mj_forward(model, data)

    def before_step(self):
        try:
            if self.sub is None:
                raise zmq.Again()
            msg = self.sub.recv(zmq.NOBLOCK)
            header = json.loads(msg[4:1284].rstrip(b"\0"))
            offset = 1284
            dtypes = {"f32": "<f4", "f64": "<f8", "i32": "<i4", "i64": "<i8", "bool": "?"}
            for field in header["fields"]:
                dtype = np.dtype(dtypes[field["dtype"]]); count = int(np.prod(field["shape"]))
                values = np.frombuffer(msg, dtype=dtype, count=count, offset=offset)
                offset += count * dtype.itemsize
                for key, short in [("action_hand_left", "lh"), ("action_hand_right", "rh")]:
                    if field["name"] == key and np.isfinite(values[-1]):
                        self.command[short] = float(np.clip(values[-1], 0, 1))
        except zmq.Again:
            pass
        for side, joints in self.jaws.items():
            for joint in joints:
                q = self.data.qpos[self.model.jnt_qposadr[joint]]
                v = self.data.qvel[self.model.jnt_dofadr[joint]]
                self.data.qfrc_applied[self.model.jnt_dofadr[joint]] = np.clip(600 * (.045 * self.command[side] - q) - 1.0 * v, -18, 18)

    def opening(self, side):
        return float(np.clip(np.mean([self.data.qpos[self.model.jnt_qposadr[j]] for j in self.jaws[side]]) / .045, 0, 1))

    def measure(self, update_score=True):
        forces = {s: [0., 0.] for s in self.pads}
        tool_forces = {s: 0. for s in self.pads}
        left_foot_force=0.
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if self.paper_task and ((contact.geom1==self.pedal_geom and contact.geom2 in self.left_foot_geoms) or
                                    (contact.geom2==self.pedal_geom and contact.geom1 in self.left_foot_geoms)):
                wrench=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,wrench)
                left_foot_force+=max(0.,float(wrench[0]))
            for side, pads in self.pads.items():
                if ((contact.geom1 in self.tool_geoms[side] and contact.geom2 in self.bottle_geoms) or
                    (contact.geom2 in self.tool_geoms[side] and contact.geom1 in self.bottle_geoms)):
                    wrench = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    tool_forces[side] += max(0., float(wrench[0]))
                for index, pad in enumerate(pads):
                    if (contact.geom1 == pad and contact.geom2 in self.bottle_geoms) or (contact.geom2 == pad and contact.geom1 in self.bottle_geoms):
                        wrench = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, i, wrench)
                        forces[side][index] += max(0., float(wrench[0]))
        lows, highs = [], []
        for geom in self.bottle_geoms:
            axis = self.data.geom_xmat[geom].reshape(3, 3)[:, 2]
            radius, halfheight = self.model.geom_size[geom, :2]
            ext = radius * np.sqrt(np.maximum(0, 1 - axis**2)) + halfheight * np.abs(axis)
            lows.append(self.data.geom_xpos[geom] - ext); highs.append(self.data.geom_xpos[geom] + ext)
        low, high = np.min(lows, axis=0), np.max(highs, axis=0)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.bottle_body, velocity, 0)
        side = max(forces, key=lambda s: min(forces[s]))
        if min(forces[side]) > .2 and self.command[side] < .8 and not self.score.grasped:
            self.active_side = side
        side = self.active_side or side
        bilateral = min(forces[side]) > .2
        tool_id = self.model.body(f"{side}_tool").id
        tcp = self.data.xpos[tool_id] + self.data.xmat[tool_id].reshape(3,3) @ np.array([.065,0,0])
        if update_score:
            self.mechanics.update(float(self.data.time), low, high, bilateral, self.opening(side), self.command[side],
                              float(np.linalg.norm(velocity[3:])), float(np.linalg.norm(velocity[:3])),
                              bool(point_aabb_distance(tcp,low,high) < .03),
                              any(force > .1 for force in tool_forces.values()))
            if self.paper_task:
                qw,qx,qy,qz=self.data.qpos[3:7]
                yaw=float(np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz)))
                self.score.update(float(self.data.time),self.data.qpos[:2],yaw,float(np.linalg.norm(self.data.qvel[:2])),
                    left_foot_force,float(self.data.qpos[self.lid_q]),float(self.data.geom_xpos[self.pedal_geom,2]-self.spec['pedal_nominal_center_z']))
        diagnostics = {}
        for tool_side in self.jaws:
            tool_id = self.model.body(f"{tool_side}_tool").id
            rotation = self.data.xmat[tool_id].reshape(3, 3)
            actual_tcp = self.data.xpos[tool_id] + rotation @ np.array([.065, 0, 0])
            wrist_tcp = self.data.site_xpos[self.model.site(f"{tool_side}_tcp").id]
            local = rotation.T @ (self.data.xpos[self.bottle_body] - actual_tcp)
            diagnostics[tool_side] = {"tcp_to_bottle_aabb_m": point_aabb_distance(actual_tcp,low,high),
                                     "pad_centers": [self.data.geom_xpos[g].tolist() for g in self.pads[tool_side]],
                                     "inner_gap_m": .006 + .09 * self.opening(tool_side),
                                     "actual_tcp": actual_tcp.tolist(), "wrist_tcp": wrist_tcp.tolist(),
                                     "mount_error_m": float(np.linalg.norm(actual_tcp-wrist_tcp)),
                                     "bottle_in_tcp_frame": local.tolist()}
        if update_score and max(tool_forces.values()) > .1 and len(self.contact_events) < 20000:
            self.contact_events.append({"sim_time": float(self.data.time), "forces": forces,
                "tool_forces": tool_forces, "geometry": diagnostics,
                "contacts": [{"geoms": [self.model.geom(int(c.geom1)).name, self.model.geom(int(c.geom2)).name],
                              "position": c.pos.tolist(), "distance": float(c.dist)}
                             for c in self.data.contact[:self.data.ncon]
                             if c.geom1 in self.bottle_geoms or c.geom2 in self.bottle_geoms]})
        self.last = {"tcp_diagnostics": diagnostics, "bottle_position": self.data.xpos[self.bottle_body].tolist(), "bottle_low": low.tolist(),
                     "bottle_high": high.tolist(), "bottle_velocity": velocity.tolist(),
                     "finger_contact_forces": forces, "gripper_commands": dict(self.command),
                     "tool_contact_forces": tool_forces,
                     "gripper_openings": {s: self.opening(s) for s in self.jaws}, **self.score.result()}
        if self.paper_task:
            self.last.update({'left_foot_pedal_force_n':left_foot_force,'lid_angle_rad':float(self.data.qpos[self.lid_q]),
                             'pedal_displacement_m':float(self.data.geom_xpos[self.pedal_geom,2]-self.spec['pedal_nominal_center_z'])})
        return self.last
