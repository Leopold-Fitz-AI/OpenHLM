"""Manage only the simulation controller started by this evaluation."""
import argparse
import os
import json
from pathlib import Path
import pty
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
source = root / "src/GR00T-WholeBodyControl4OpenHLM/gear_sonic_deploy"
run = args.run_dir.resolve()
run.mkdir(parents=True, exist_ok=True)
master, slave = pty.openpty()
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = ":".join([
    str(root / "native/tensorrt/lib"), str(root / "native/python/tensorrt_libs"),
    str(root / "native/ortpython/onnxruntime/capi"),
    str(root / "native/onnxruntime/lib"),
    str(root / "native/cyclonedds/lib"), "/usr/local/cuda/lib64",
    env.get("LD_LIBRARY_PATH", ""),
])
preview=float(env.get("OPENHLM_PREVIEW_SECONDS", ".2"))
if not 0 <= preview <= .6:raise ValueError("Preview must be 0..0.6 seconds")
(run / "controller_config.json").write_text(json.dumps({"preview_seconds":preview,"controller_period_s":.02,
    "dds_domain":int(env.get("OPENHLM_DDS_DOMAIN","0")),
    "right_arm_reference_servo":env.get("OPENHLM_ARM_REFERENCE_SERVO","0")=="1",
    "source_preview_window":env.get("OPENHLM_SOURCE_PREVIEW","0")=="1"},indent=2))
command = [str(source / "target/release/g1_deploy_onnx_ref"), "lo",
           "policy/release/model_decoder.onnx", "reference/example/",
           "--obs-config", "policy/release/observation_config.yaml",
           "--encoder-file", "policy/release/model_encoder.onnx",
           "--input-type", "zmq", "--output-type", "zmq",
           "--zmq-host", "127.0.0.1", "--zmq-port", "20556",
           "--zmq-out-port", "20557", "--disable-crc-check",
           "--logs-dir", str(run / "control_logs")]
with (run / "controller.log").open("wb") as log:
    child = subprocess.Popen(command, cwd=source, env=env, stdin=slave,
                             stdout=log, stderr=subprocess.STDOUT)
    os.close(slave)
    (run / "controller.pid").write_text(str(child.pid))
    sent = set()
    try:
        while child.poll() is None:
            for filename, key in [("controller-start", b"]"), ("controller-stream", b"\n")]:
                if filename not in sent and (run / filename).exists():
                    os.write(master, key)
                    sent.add(filename)
            if (run / "stop").exists():
                os.write(master, b"O")
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=10)
                break
            time.sleep(0.1)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
        os.close(master)
    (run / "controller-exit.txt").write_text(str(child.returncode))
    print("Controller exit:", child.returncode, flush=True)
