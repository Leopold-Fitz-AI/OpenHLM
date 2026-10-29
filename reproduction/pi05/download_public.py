"""Download only official public OpenHLM assets into the reproduction directory."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

from huggingface_hub import HfApi, snapshot_download

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--only", choices=["model", "dataset", "all"], default="all")
args = parser.parse_args()
api = HfApi()
for repo, kind, patterns, destination in [
    ("OpenHLM/OpenHLM-ckpts", "model", ["12tasks_12full-teleop/*"], root / "models"),
    ("OpenHLM/OpenHLM-data", "dataset", ["example_demonstration/*"], root / "public_data"),
]:
    if args.only not in ("all", kind):
        continue
    info = api.repo_info(repo_id=repo, repo_type=kind)
    print(f"Downloading {repo}@{info.sha}", flush=True)
    for attempt in range(3):
        try:
            snapshot_download(
                repo_id=repo, repo_type=kind, revision=info.sha,
                allow_patterns=patterns, local_dir=destination, max_workers=2,
            )
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(5)
    (destination / "source_revision.json").write_text(json.dumps({
        "repo_id": repo, "revision": info.sha, "patterns": patterns,
    }, indent=2) + "\n")
