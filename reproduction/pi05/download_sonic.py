import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
from huggingface_hub import HfApi, hf_hub_download

root = Path(__file__).resolve().parents[2]
destination = root / "src/GR00T-WholeBodyControl4OpenHLM/gear_sonic_deploy/policy/release"
revision = HfApi().model_info("nvidia/GEAR-SONIC").sha
for name in ["model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"]:
    source = hf_hub_download("nvidia/GEAR-SONIC", name, revision=revision)
    shutil.copyfile(source, destination / name)
    print(name, flush=True)
(root / "logs/sonic-model-source.json").write_text(json.dumps({
    "repo": "nvidia/GEAR-SONIC", "revision": revision,
    "files": ["model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"],
}, indent=2))
