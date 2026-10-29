#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export HF_LEROBOT_HOME="$ROOT/lerobot_home" WANDB_MODE=disabled
cd "$ROOT/src/openpi4OpenHLM"
exec "$ROOT/.venv/bin/python" scripts/train_pytorch.py openhlm_example \
  --exp-name "${EXPERIMENT_NAME:-pi05_cpu_base_smoke}" \
  --batch-size 1 --num-workers 0 --num-train-steps 2 \
  --save-interval 2 --log-interval 1 --eval-interval 0 \
  --no-wandb-enabled --no-enable-training-compile \
  --no-enable-gradient-checkpointing \
  --pytorch-weight-path "${BASE_CHECKPOINT:-/root/models/pi05/pi05_base_pytorch}"
