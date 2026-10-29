"""Headless G1 physics and camera bridge for bounded, simulation-only evaluation."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

import cv2
import msgpack
import mujoco
import numpy as np
import zmq
from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--duration", type=float, default=30)
parser.add_argument("--task", choices=["none", "bottle-bin", "paper-disposal"], default="none")
parser.add_argument("--layout-id",type=int,default=0)
parser.add_argument("--build-only",action='store_true')
parser.add_argument("--camera-profile", choices=["pinhole", "approximate-fisheye"], default="pinhole")
parser.add_argument("--tool-profile", choices=["legacy", "visual-fit"], default="legacy")
parser.add_argument("--run-label", choices=["vla","baseline","replay","calibration","tracking-probe","scripted-baseline"], default="vla")
parser.add_argument("--wrist-calibration",type=Path)
args = parser.parse_args()
run = args.run_dir.resolve()
run.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[2]
assets = root / "src/GR00T-WholeBodyControl4OpenHLM/gear_sonic/data/robot_model/model_data/g1"
robot = ET.parse(assets / "g1_29dof_with_hand.xml")
robot.getroot().find("compiler").set("meshdir", str(assets / "meshes"))
for side in ["left", "right"]:
    body = robot.find(f".//body[@name='{side}_wrist_yaw_link']")
    ET.SubElement(body, "camera", name=f"eval_{side}_wrist", pos="0.09 0 0.08",
                  xyaxes="0 -1 0 0.9 0 0.435", fovy="75")
if args.tool_profile == "visual-fit":
    fit = json.loads((args.wrist_calibration or Path(__file__).with_name("wrist_camera_visual_fit.json")).read_text())
    (run / "camera_fit.json").write_text(json.dumps(fit,indent=2))
    for side, camera in fit["cameras"].items():
        node=robot.find(f".//camera[@name='eval_{side}_wrist']")
        node.set("pos", " ".join(map(str,camera["position"])))
        node.set("xyaxes", " ".join(map(str,camera["xyaxes"])))
        node.set("fovy", str(camera["fovy"]))
robot.write(run / "robot.xml")
scene = ET.parse(assets / "scene_43dof.xml")
ET.SubElement(scene.getroot(), "option", integrator="implicitfast", cone="elliptic",
              impratio="10", noslip_iterations="5", iterations="80")
scene.getroot().find("include").set("file", str(run / "robot.xml"))
if args.task == "bottle-bin":
    from bottle_task import build_task
    build_task(robot, scene, run, tool_profile=args.tool_profile)
elif args.task == "paper-disposal":
    from paper_disposal_scene import build_paper_task
    build_paper_task(robot,scene,run,tool_profile=args.tool_profile,layout_id=args.layout_id)
scene.write(run / "scene.xml")
if args.build_only:
    compiled=mujoco.MjModel.from_xml_path(str(run/'scene.xml'))
    print(json.dumps({'scene':str(run/'scene.xml'),'nq':compiled.nq,'nv':compiled.nv,
        'nu':compiled.nu,'layout_id':args.layout_id,'compiled':True}))
    raise SystemExit(0)
config = SimLoopConfig(enable_onscreen=False, enable_offscreen=False, sim_frequency=250).load_wbc_yaml()
config["ROBOT_SCENE"] = str(run / "scene.xml")
config["DOMAIN_ID"] = int(os.environ.get("OPENHLM_DDS_DOMAIN", "0"))
sim = BaseSimulator(config=config, onscreen=False, offscreen=False)
env = sim.sim_env
env.elastic_band.point = env.elastic_band.point.astype(np.float64)
model, data = env.mj_model, env.mj_data
ctx = zmq.Context()
task = None
if args.task in ["bottle-bin","paper-disposal"]:
    from bottle_task import BottleTaskPhysics
    task = BottleTaskPhysics(model, data, ctx,spec=json.loads((run/'task_spec.json').read_text()))
render_log=(run / "render.log").open("wb")
render_process=subprocess.Popen([sys.executable,str(Path(__file__).with_name("g1_render_worker.py")),
    "--run-dir",str(run),"--camera-profile",args.camera_profile,"--run-label",args.run_label],
    stdout=render_log,stderr=subprocess.STDOUT)
rows = []
video_frames = 0
fallen = [False]
def check_fall():
    if data.qpos[2] < 0.2:
        fallen[0] = True
env.check_fall = check_fall  # Record the first fall rather than silently resetting it.
(run / "sim-ready").touch()
started = time.monotonic()
released = None
release_requested = None
evaluation_started = None
next_render = 0.0
next_video = 0.0
max_physics_backlog = 0.0
report = {}
task_rows = []
state_qpos, state_qvel, state_times = [], [], []
# Rendering uses an independent model/data snapshot. Its GL work must never
# stall DDS/physics or mutate the physical solver's warm-start state.
physics_lock = threading.Lock()
physics_stop = threading.Event()
physics_errors = []
step_durations=[]
next_task_measure=0.
warmup_resyncs=0
def physics_loop():
    global max_physics_backlog,warmup_resyncs,next_task_measure
    clock = time.monotonic()
    accumulator = 0.0
    try:
        while not physics_stop.is_set() and not fallen[0]:
            now = time.monotonic()
            accumulator += now - clock
            clock = now
            max_physics_backlog = max(max_physics_backlog, accumulator)
            if accumulator > 0.1:
                if evaluation_started is not None:
                    raise RuntimeError("Physics backlog exceeded 0.1s during evaluation")
                # GPU/GL initialization is outside the trial. Do not integrate
                # a burst of stale wall time while the startup support is on.
                accumulator=config["SIMULATE_DT"]
                warmup_resyncs+=1
            while accumulator >= config["SIMULATE_DT"]:
                with physics_lock:
                    # Publish sensors while waiting for the controller; letting
                    # unpowered joints fall during TensorRT startup gives each
                    # trial a different, sometimes unrecoverable initial pose.
                    # This ends before support lowering and all task scoring.
                    if not (run / "controller-stream").exists():
                        env.unitree_bridge.PublishLowState(env.prepare_obs())
                        accumulator -= config["SIMULATE_DT"]
                        continue
                    if task: task.before_step()
                    step_start=time.monotonic()
                    env.sim_step()
                    if task and data.time>=next_task_measure:
                        task.measure(update_score=evaluation_started is not None)
                        next_task_measure+=.02
                        if task.score.success: (run / "task-success").touch()
                    step_durations.append(time.monotonic()-step_start)
                accumulator -= config["SIMULATE_DT"]
                if fallen[0]: break
            physics_stop.wait(max(0, config["SIMULATE_DT"] - accumulator))
    except Exception as exc:
        physics_errors.append(exc)
physics_thread = threading.Thread(target=physics_loop, name="g1-physics", daemon=True)
physics_thread.start()
try:
    while time.monotonic() - started < 900:
        tick = time.monotonic()
        if release_requested is None and (run / "release").exists():
            release_requested = tick
        if release_requested is not None and released is None:
            fraction = min(1.0, (tick - release_requested) / 2.0)
            env.elastic_band.point[2] = 1.0 - 0.22 * fraction
            if fraction >= 1.0:
                env.elastic_band.enable = False
                released = tick
                print("Virtual support lowered over 2s, then released", flush=True)
        placement_file = run / "place-standing.json"
        if placement_file.exists() and not (run / "placement-complete").exists():
            if evaluation_started is not None:
                raise RuntimeError("Standing placement is only allowed before evaluation")
            requested=json.loads(placement_file.read_text())
            with physics_lock:
                before=data.qpos[:7].copy()
                delta=np.array([requested["x"],requested["y"]])-data.qpos[:2]
                data.qpos[:2]+=delta
                if task:
                    for side in ["lh","rh"]:
                        address=model.jnt_qposadr[model.joint(f"{side}_tool_free").id]
                        data.qpos[address:address+2]+=delta
                mujoco.mj_forward(model,data)
                after=data.qpos[:7].copy()
            (run / "initial_placement.json").write_text(json.dumps({"before":before.tolist(),"after":after.tolist(),
                "delta_xy":delta.tolist(),"before_evaluation":True,"objects_moved":False},indent=2))
            (run / "placement-complete").touch()
        if evaluation_started is None and (run / "evaluation-start").exists():
            evaluation_started = tick
        if (run / "stop").exists() or (released and tick - released > args.duration):
            break
        if physics_errors:
            raise RuntimeError("Physics worker failed") from physics_errors[0]
        if not np.isfinite(data.qpos).all() or fallen[0]:
            break
        if tick >= next_render:
            with physics_lock:
                qpos=data.qpos.copy();qvel=data.qvel.copy();sim_time=float(data.time)
            state_qpos.append(qpos);state_qvel.append(qvel)
            state_times.append([tick-started,sim_time,bool(evaluation_started)])
            tilt=float(np.arccos(np.clip(1-2*(qpos[4]**2+qpos[5]**2),-1,1)))
            status={"height":float(qpos[2]),"tilt_radians":tilt,"qpos":qpos.tolist(),
                    "qvel":qvel.tolist(),"sim_time":sim_time,"released":released is not None,
                    "evaluating":evaluation_started is not None,"monotonic_time_s":tick}
            if task:
                status["task"]=task.last
                task_rows.append({"sim_time":sim_time,"wall_time":tick-started,
                                  "evaluating":bool(evaluation_started),**task.last})
            (run / "status.tmp").write_text(json.dumps(status))
            (run / "status.tmp").replace(run / "status.json")
            rows.append([tick-started,sim_time,float(qpos[2]),*qpos[3:7].tolist(),bool(released),bool(evaluation_started)])
            next_render=tick+1/30
        if render_process.poll() is not None:raise RuntimeError("Camera process exited; inspect render.log")
        time.sleep(max(0, config["SIMULATE_DT"] - (time.monotonic() - tick)))
finally:
    physics_stop.set()
    physics_thread.join(timeout=5)
    (run / "render-stop").touch()
    try:render_process.wait(timeout=15)
    except subprocess.TimeoutExpired:render_process.terminate();render_process.wait(timeout=10)
    render_log.close()
    render_report=json.loads((run / "render_report.json").read_text()) if (run / "render_report.json").exists() else {}
    heights = [r[2] for r in rows if r[7]]
    evaluation_heights = [r[2] for r in rows if r[8]]
    report = {"run_label": args.run_label, "camera_profile": args.camera_profile, "tool_profile": args.tool_profile, "max_physics_backlog_s": max_physics_backlog,
              "physics_step_s":config["SIMULATE_DT"],"integrator":"implicitfast","cone":"elliptic","impratio":10,"noslip_iterations":5,
              "initialization_clock_resyncs":warmup_resyncs,
              "physics_compute_percentiles_s":np.percentile(step_durations,[50,95,99,100]).tolist() if step_durations else [],
              "camera_target_hz": 30,
              "camera_actual_hz":render_report.get("camera_actual_hz",0),"separate_render_process":True,"contact_scoring_hz":50,
              "independent_physics_worker": True, "physics_worker_error": str(physics_errors[0]) if physics_errors else None, "wrist_mount_profile": args.tool_profile, "released": released is not None, "fallen": fallen[0],
              "released_wall_seconds": time.monotonic() - released if released else 0,
              "finite_state": bool(np.isfinite(data.qpos).all()),
              "min_released_height": min(heights) if heights else None,
              "evaluation_started": evaluation_started is not None,
              "evaluated_wall_seconds": time.monotonic() - evaluation_started if evaluation_started else 0,
              "min_evaluation_height": min(evaluation_heights) if evaluation_heights else None,
              "fall_phase": ("evaluation" if evaluation_started else "settle" if released else "lowering" if release_requested else "warmup") if fallen[0] else None,
              "final_height": float(data.qpos[2]), "rendered_frames": len(rows),
              "video_frames":render_report.get("video_frames",0),
              "support_lowering_seconds": 2,
              "limits": ["empty-floor scene; no manipulation success metric",
                         "added wrist camera mounts approximate the hardware",
                         "CTAG gripper state represented by an open-state stub"]}
    if task:
        report["task"] = task.score.result()
        report["task_name"] = task.spec.get('task_id','green_bottle_to_bin')
        report['task_protocol']=task.spec.get('protocol','legacy_open_bin_diagnostic')
        report["limits"] = ["parallel-jaw physical surrogate, not calibrated CTAG hardware",
                            "synthetic scene and approximate camera intrinsics/extrinsics"]
        (run / "task_telemetry.json").write_text(json.dumps(task_rows))
        (run / "contact_events.json").write_text(json.dumps(task.contact_events))
    if state_qpos:
        np.savez_compressed(run / "physics_trace.npz", qpos=np.stack(state_qpos),
                            qvel=np.stack(state_qvel), times=np.asarray(state_times),monotonic_time_s=np.asarray(state_times)[:,0]+started)
    (run / "simulation.json").write_text(json.dumps(report, indent=2))
    with (run / "telemetry.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["wall_time", "sim_time", "height", "qw", "qx", "qy", "qz", "released", "evaluating"])
        writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)
    sim.close()
    if task:
        task.sub.close(linger=0)
    ctx.term()
