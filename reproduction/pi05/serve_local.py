"""Serve the verified policy on loopback; use SSH forwarding for remote clients."""
import argparse
import logging
from pathlib import Path

import torch
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--port", type=int, default=18005)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)
torch.set_num_threads(16)
policy = policy_config.create_trained_policy(
    config.get_config("openhlm_example"), args.checkpoint,
    pytorch_device=args.device, sample_kwargs={"num_steps": 10},
)
WebsocketPolicyServer(
    policy=policy, host="127.0.0.1", port=args.port, metadata=policy.metadata,
).serve_forever()
