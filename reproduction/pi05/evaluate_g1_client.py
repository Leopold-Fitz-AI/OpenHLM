"""Exercise the repository's G1 client against its original C++ WBC in simulation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
from openpi_client import image_tools, websocket_client_policy
from pose_stream import TimedPoseStream
from arm_tracking import ArmBiasCompensator

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src/GR00T-WholeBodyControl4OpenHLM/openpi-eval"))
sys.path.insert(0, str(root / "src/GR00T-WholeBodyControl4OpenHLM"))
from sonic_g1_env import SonicG1Env, DEFAULT_MIMIC_OBS_G1,DEFAULT_MIMIC_OBS_G1_RAISED
from preview_stream import PreviewWindowStream,ProbeTrajectory

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--mode", choices=["baseline", "vla", "replay", "calibration", "tracking-probe", "scripted-baseline"], default="baseline")
parser.add_argument("--seconds", type=float, default=20)
parser.add_argument("--start-x",type=float,default=0.)
parser.add_argument("--execution", choices=["synchronous", "async", "hold"], default="synchronous")
parser.add_argument("--replay-run", type=Path)
parser.add_argument("--transport-hz", type=int, default=30)
parser.add_argument("--standardize-start",action="store_true")
parser.add_argument("--arm-bias-compensation",action="store_true")
parser.add_argument("--future-window",action="store_true")
parser.add_argument("--initial-pose",choices=['lowered','raised'],default='lowered')
parser.add_argument("--warmup-queries",type=int,default=2)
parser.add_argument("--instruction")
args = parser.parse_args()
future_vla=args.future_window and args.mode=="vla"
if future_vla and args.execution!="hold":parser.error("VLA future preview currently uses --execution hold")
replay_actions=None
if args.mode == "replay":
    if args.replay_run is None: raise ValueError("Replay requires --replay-run")
    c=json.loads((args.replay_run/"client.json").read_text())
    chunks=np.load(args.replay_run/"policy_trace.npz")["actions"]
    repl=json.loads((args.replay_run/"replans.json").read_text())
    replay_actions=np.empty((c["executed_steps"],34))
    starts=[0]+[x["apply_step"] for x in repl]
    skips=[0]+[x["skipped_predictions"] for x in repl]
    for i,(a,b) in enumerate(zip(starts,starts[1:]+[len(replay_actions)])):
        replay_actions[a:b]=chunks[i,skips[i]:skips[i]+b-a]
run = args.run_dir.resolve()
deadline = time.monotonic() + 600
while not (run / "controller.log").exists() or "G1Deploy object created successfully" not in (run / "controller.log").read_text(errors="replace"):
    if (run / "controller-exit.txt").exists() or time.monotonic() > deadline:
        raise RuntimeError("Controller did not initialize; inspect controller.log")
    time.sleep(0.5)
env = SonicG1Env(control_hz=args.transport_hz, vision_server_address="127.0.0.1", vision_server_port=20555,
                 wrist_server_port=20554, action_zmq_address="127.0.0.1", action_zmq_port=20556,
                 body_state_address="127.0.0.1", body_state_port=20557,
                 hand_state_address="127.0.0.1", hand_state_port=20558)
env.vision_client.image_shape = None
env.wrist_client.image_shape = None
task_spec=json.loads((run/'task_spec.json').read_text()) if (run/'task_spec.json').exists() else {}
from paper_protocol import PAPER_PROMPT
instruction=args.instruction or task_spec.get('paper_prompt',PAPER_PROMPT)
pose = DEFAULT_MIMIC_OBS_G1_RAISED if args.initial_pose=='raised' else DEFAULT_MIMIC_OBS_G1
default = np.concatenate([pose[18:25], [1.], pose[25:32], [1.], pose[3:18], pose[:3]]).astype(np.float32)
latencies, steps = [], 0
executor = ThreadPoolExecutor(max_workers=1)
replans, observations, predicted = [], [], []
observation_ages = []
hold_steps = 0
executed_actions, measured_states, control_times = [], [], []
sender = None
arm_tracker = ArmBiasCompensator() if args.arm_bias_compensation else None
last_control_stamp = None
bias_trace=[]
wbc_debug=[]
wbc_debug_times=[]
started = None
try:
    (run / "controller-start").touch()
    until = time.monotonic() + 15
    while time.monotonic() < until:
        obs = env.get_observation()
        if env._latest_body_state is not None and obs["head_image_left"].any():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Missing actual simulated body state or camera frames")
    env.reset()
    if args.initial_pose=='raised':env.reset(default_pose=DEFAULT_MIMIC_OBS_G1_RAISED)
    for i in range(90):
        env.step(default)
        if i == 10:
            (run / "controller-stream").touch()
        time.sleep(1 / 30)
    policy = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 18006) if args.mode == "vla" else None
    def request(obs, last_action=None):
        data = {"observation/head_image_left": image_tools.resize_with_pad(obs["head_image_left"], 224, 224),
                "observation/left_wrist_image": image_tools.resize_with_pad(obs["left_wrist_image"], 224, 224),
                "observation/right_wrist_image": image_tools.resize_with_pad(obs["right_wrist_image"], 224, 224),
                "observation/state": obs["state"],
                "prompt": instruction}
        if last_action is not None:
            data["observation/last_action"] = last_action
        begin = time.monotonic()
        result = policy.infer(data)["actions"]
        latencies.append(time.monotonic() - begin)
        assert result.shape == (50, 34) and np.isfinite(result).all()
        observations.append(data["observation/state"].copy())
        predicted.append(result.copy())
        return result
    if policy:
        for _ in range(args.warmup_queries):
            request(env.get_observation())
        latencies.clear()
        observations.clear(); predicted.clear()
    # Warmup requests must not determine the starting state of the robot trial.
    for _ in range(60):
        env.step(default)
        time.sleep(1 / 30)
    (run / "release").touch()
    # Lower the virtual support for 2s, then allow 3s of unsupported settling.
    for _ in range(150):
        env.step(default)
        time.sleep(1 / 30)
    status = json.loads((run / "status.json").read_text())
    if status["height"] < 0.6 or status["tilt_radians"] > 0.4:
        raise RuntimeError(f"Invalid standing initialization: {status}")
    if args.standardize_start:
        (run / "place-standing.json").write_text(json.dumps({"x":args.start_x,"y":0.}))
        until=time.monotonic()+5
        while not (run / "placement-complete").exists():
            if time.monotonic()>until:raise RuntimeError("Standing placement timed out")
            env.step(default);time.sleep(1/30)
        for _ in range(30):env.step(default);time.sleep(1/30)
        status=json.loads((run / "status.json").read_text())
        if status["height"]<.6 or status["tilt_radians"]>.4 or np.linalg.norm(np.array(status["qpos"][:2])-[args.start_x,0])>.06:
            raise RuntimeError("Standing placement did not remain stable")
    (run / "initial_state.json").write_text(json.dumps(status,indent=2))
    calibration=None
    if args.mode == "scripted-baseline":
        from g1_scripted_baseline import G1ScriptedBaseline
        calibration=G1ScriptedBaseline(run,default,status)
    if args.mode == "calibration":
        from g1_tcp_calibration import G1TCPCalibration
        calibration=G1TCPCalibration(run,default,status)
    if args.future_window:
        if args.mode not in ["tracking-probe","scripted-baseline","vla"]:raise ValueError("Invalid future-window source")
        if future_vla:
            from action_chunk_trajectory import ActionChunkTrajectory
            trajectory=ActionChunkTrajectory(default,initial_yaw=env._accumulated_yaw)
        else:trajectory=calibration if args.mode == "scripted-baseline" else ProbeTrajectory(default)
        sender=PreviewWindowStream(env,trajectory,hz=50,lookahead_s=.2)
        sender.start()
    elif args.transport_hz != 30:
        sender=TimedPoseStream(env.step,default,hz=args.transport_hz)
        sender.start()
    (run / "evaluation-start").touch()
    started = time.monotonic()
    chunk, index, last = None, 0, None
    pending, pending_step, last_swap_step = None, None, 0
    while time.monotonic() - started < args.seconds:
        tick = time.monotonic()
        if (run / "simulation.json").exists() or (run / "task-success").exists():
            break
        obs = env.get_observation()
        ages = env.observation_ages()
        # Scripts use physical feedback; VLA requires fresh RGB and hand state.
        required_ages=ages if args.mode=="vla" else {"body":ages["body"]}
        if max(required_ages.values()) > 0.5:
            raise RuntimeError(f"Stale or missing observation stream: {ages}")
        observation_ages.append(ages)
        if policy:
            if chunk is None or (args.execution == "synchronous" and index >= 25):
                chunk, index = request(obs, last), 0
                if future_vla:trajectory.replace(chunk)
                if steps == 0:
                    np.save(run / "first_action_chunk.npy", chunk)
                    np.savez_compressed(run / "first_observation.npz", **obs)
            if future_vla:index=trajectory.index()
            if args.execution == "async":
                # Prefetch at step 12, replace after 25 control ticks. Discard
                # predictions for ticks elapsed since the new observation.
                if pending is not None and pending.done() and steps >= last_swap_step + 25:
                    chunk = pending.result()
                    index = steps - pending_step
                    replans.append({"capture_step": pending_step, "apply_step": steps,
                                    "skipped_predictions": index})
                    pending = None
                    last_swap_step = steps
                if pending is None and steps >= last_swap_step + 12:
                    pending_step = steps
                    pending = executor.submit(request, obs, last)
                if index >= len(chunk):
                    raise RuntimeError("Policy inference exceeded the buffered action horizon")
            if args.execution == "hold":
                if pending is not None and pending.done():
                    chunk, index = pending.result(), 0
                    if future_vla:trajectory.replace(chunk)
                    replans.append({"capture_step":pending_step,"apply_step":steps,
                                    "skipped_predictions":0,"held_ticks":steps-pending_step})
                    pending = None
                if index >= 25 and pending is None:
                    pending_step = steps
                    pending = executor.submit(request,obs,last)
        holding = policy is not None and args.execution == "hold" and pending is not None
        action = last.copy() if holding else chunk[index] if policy else default
        if future_vla:action=trajectory.window([time.monotonic()])[0][0]
        if args.mode == "tracking-probe":
            elapsed_probe=time.monotonic()-started
            action=default.copy()
            # Slow, bounded joints at rest, away from the bottle.
            phase=int(elapsed_probe//12)
            joint=[8,11,13][min(phase,2)]
            local_t=elapsed_probe%12
            action[joint]+=.12*np.sin(2*np.pi*local_t/6)
        debug=env._latest_body_state
        if debug and all(k in debug for k in ["body_q_target","body_q_measured","last_action"]):
            wbc_debug.append([debug["body_q_target"],debug["last_action"],debug["body_q_measured"]])
            wbc_debug_times.append(time.monotonic())
        if calibration:
            status=json.loads((run / "status.json").read_text())
            if time.monotonic()-status["monotonic_time_s"]>.25:
                raise RuntimeError("Stale physical feedback for scripted controller")
            action=calibration.action(status,time.monotonic()-started)
        if replay_actions is not None:
            if steps >= len(replay_actions): break
            action=replay_actions[steps]
        if holding:
            # Hold a pose during inference; do not integrate a held yaw rate.
            action[33] = 0
            hold_steps += 1
        reference=action
        if arm_tracker:
            stamp=time.monotonic()
            reference=arm_tracker.update(action,obs["state"],stamp-last_control_stamp if last_control_stamp else 1/30)
            last_control_stamp=stamp
            bias_trace.append(arm_tracker.bias.copy())
        if sender:
            sender.check();sender.submit(reference)
        else:
            env.step(reference)
        control_times.append(time.monotonic())
        executed_actions.append(np.array(action,copy=True))
        measured_states.append(obs["state"].copy())
        last = action.copy()
        if not holding: index += 1
        steps += 1
        time.sleep(max(0, 1 / 30 - (time.monotonic() - tick)))
    if (run / "task-success").exists():
        # Record the physical result and open jaws for two more seconds.
        time.sleep(2.)
finally:
    elapsed = time.monotonic() - started if started else 0
    (run / "stop").touch()
    if sender: sender.close()
    executor.shutdown(wait=True, cancel_futures=True)
    env.close()
    (run / "stop").touch()
    if executed_actions:
        np.savez_compressed(run / "control_trace.npz", actions=np.stack(executed_actions),
                            measured_states=np.stack(measured_states),monotonic_time_s=np.asarray(control_times))
    if wbc_debug:
        np.savez_compressed(run / "wbc_debug_trace.npz",q=np.asarray(wbc_debug),
                            monotonic_time_s=np.asarray(wbc_debug_times),
                            channels=np.array(["reference","motor_target","measured"]))
    if bias_trace:
        np.save(run / "arm_bias_trace.npy",np.stack(bias_trace))
    if sender and sender.actions:
        np.savez_compressed(run / "transport_trace.npz",actions=np.stack(sender.actions),
                            monotonic_time_s=np.asarray(sender.times))
    if observations:
        np.savez_compressed(run / "policy_trace.npz", states=np.stack(observations),
                            actions=np.stack(predicted))
    (run / "replans.json").write_text(json.dumps(replans, indent=2))
    report = {"future_window": args.future_window, "rgb_required_for_control":args.mode=="vla", "arm_bias_compensation": args.arm_bias_compensation, "standardized_start":args.standardize_start, "mode": args.mode, "privileged_calibration_not_vla": args.mode in ["calibration","scripted-baseline"], "execution": args.execution,
              "instruction":instruction,"initial_pose":args.initial_pose,"warmup_queries":args.warmup_queries,
              "transport_target_hz": 50 if args.future_window else args.transport_hz,
              "working_start_x_m": args.start_x if args.standardize_start else None,
              "right_arm_reference_servo":json.loads((run/"controller_config.json").read_text()).get("right_arm_reference_servo",False),
              "transport_actual_hz": (len(sender.times)-1)/(sender.times[-1]-sender.times[0]) if sender and len(sender.times)>1 else None,
              "max_receive_age_seconds": {k:max(x[k] for x in observation_ages) for k in observation_ages[0]} if observation_ages else {}, "executed_steps": steps, "elapsed_seconds": elapsed,
              "effective_hz": steps / elapsed if elapsed else 0, "inference_hold_steps": hold_steps,
              "policy_queries": len(latencies), "policy_seconds": latencies,
              "settle_seconds_before_evaluation": 3,
              "support_lowering_seconds": 2,
              "integration_path": "OpenHLM -> SonicG1Env -> ZMQ -> C++ SONIC (optional arm servo recorded in controller_config.json) -> DDS -> MuJoCo",
              "task_success_evaluated": (run / "task_spec.json").exists()}
    (run / "client.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
