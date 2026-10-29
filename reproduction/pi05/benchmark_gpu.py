"""Measure warmed-up GPU policy latency; synthetic observations, not task success."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from openpi.policies import g1_policy, policy_config
from openpi.training import config

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repeats", type=int, default=10)
args = parser.parse_args()
torch.set_num_threads(8)
torch.manual_seed(42)
np.random.seed(42)
policy = policy_config.create_trained_policy(
    config.get_config("openhlm_example"), args.checkpoint,
    pytorch_device="cuda", sample_kwargs={"num_steps": 10},
)
obs = g1_policy.make_g1_example()
for _ in range(2):
    policy.infer(obs)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
timings = []
for _ in range(args.repeats):
    start = time.perf_counter()
    actions = policy.infer(obs)["actions"]
    torch.cuda.synchronize()
    timings.append(1000 * (time.perf_counter() - start))
    assert actions.shape == (50, 34) and np.isfinite(actions).all()
report = {
    "device": torch.cuda.get_device_name(), "torch": torch.__version__,
    "sampling_steps": 10, "repeats": args.repeats,
    "latency_ms": {"mean": float(np.mean(timings)),
                   "p50": float(np.median(timings)), "p95": float(np.percentile(timings, 95)),
                   "samples": timings},
    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    "shape": [50, 34], "all_finite": True,
    "compile_disabled": __import__("os").environ.get("TORCH_COMPILE_DISABLE"),
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
