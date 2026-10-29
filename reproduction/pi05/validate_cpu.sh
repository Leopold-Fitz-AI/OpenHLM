#!/usr/bin/env bash
# Run inside the dedicated server environment. Does not use occupied GPUs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu TORCH_COMPILE_DISABLE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export HF_LEROBOT_HOME="$ROOT/lerobot_home"
export WANDB_MODE=disabled
PYTHON="$ROOT/.venv/bin/python"
cd "$ROOT/src/openpi4OpenHLM"
"$PYTHON" "$ROOT/reproduction/pi05/smoke_inference.py" \
  --checkpoint "$ROOT/models/12tasks_12full-teleop" \
  --output "$ROOT/logs/inference_cpu.json"
if [ ! -f "$HF_LEROBOT_HOME/OpenHLM/example/meta/info.json" ]; then
  "$PYTHON" examples/unitree_g1/convert_g1_data_to_lerobot_multi.py \
    --parent-dir "$ROOT/public_data/example_demonstration" \
    --dataset-folders 20260520_1602_example_1 20260520_1602_example_2 \
    --repo-name OpenHLM/example
fi
"$PYTHON" scripts/compute_norm_stats.py --config-name openhlm_example
bash "$ROOT/reproduction/pi05/train_cpu.sh"
echo 'CPU validation completed: inference, public data conversion, statistics, 2 training steps.'
