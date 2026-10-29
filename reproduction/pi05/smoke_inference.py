"""Validate official OpenHLM weights without sending commands to a robot."""
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
parser.add_argument("--device", default="cpu")
parser.add_argument("--server-port", type=int)
args = parser.parse_args()
torch.set_num_threads(16)
torch.manual_seed(42)
np.random.seed(42)
started = time.monotonic()
if args.server_port is None:
    policy = policy_config.create_trained_policy(
        config.get_config("openhlm_example"), args.checkpoint,
        pytorch_device=args.device, sample_kwargs={"num_steps": 10},
    )
else:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    policy = WebsocketClientPolicy(host="127.0.0.1", port=args.server_port)
example = g1_policy.make_g1_example()
example["prompt"] = "pick up the fruit"
result = policy.infer(example)
actions = result["actions"]
assert actions.shape == (50, 34), actions.shape
assert np.isfinite(actions).all(), "Non-finite model output"
report = {
    "checkpoint": str(args.checkpoint.resolve()), "device": args.device,
    "torch": torch.__version__, "shape": list(actions.shape),
    "server_port": args.server_port,
    "all_finite": bool(np.isfinite(actions).all()),
    "elapsed_seconds": time.monotonic() - started,
    "policy_timing": result["policy_timing"],
    "observation": "synthetic; validates execution, not task success",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
np.save(args.output.with_suffix(".actions.npy"), actions)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
