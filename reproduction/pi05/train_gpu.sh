#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to explicitly allocated GPU indices}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Use numeric GPU indices" >&2; exit 1
  fi
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
  if (( used > 2000 )); then
    echo "GPU $gpu has ${used}MiB allocated; refusing to overlap another task." >&2
    exit 1
  fi
done
export HF_LEROBOT_HOME="$ROOT/lerobot_home" JAX_PLATFORMS=cpu
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled
cd "$ROOT/src/openpi4OpenHLM"
exec "$ROOT/.venv/bin/python" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${#GPUS[@]}" \
  scripts/train_pytorch.py openhlm_example \
  --exp-name "${EXPERIMENT_NAME:-pi05_example_train}" \
  --batch-size "${BATCH_SIZE:-8}" --num-workers 4 \
  --num-train-steps "${TRAIN_STEPS:-30000}" \
  --no-wandb-enabled --enable-gradient-checkpointing \
  --pytorch-weight-path "${BASE_CHECKPOINT:-/root/models/pi05/pi05_base_pytorch}" "$@"
