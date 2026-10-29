#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
deadline=$((SECONDS + 3600))
until [ -f models/source_revision.json ] && [ -f public_data/source_revision.json ] \
  && .venv/bin/python -c 'import torch; assert torch.__version__.startswith("2.9.0")' >/dev/null 2>&1; do
  if (( SECONDS > deadline )); then
    echo 'Timed out waiting for downloads/dependencies' >&2; exit 1
  fi
  sleep 15
done
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=16
/root/.local/bin/uv pip check --python .venv/bin/python
/root/.local/bin/uv pip freeze --python .venv/bin/python > logs/environment-final.txt
sha256sum models/12tasks_12full-teleop/model.safetensors > logs/checkpoint.sha256
cd src/openpi4OpenHLM
"$ROOT/.venv/bin/python" -m pytest -q src/openpi/transforms_test.py src/openpi/shared/normalize_test.py
cd "$ROOT"
bash reproduction/pi05/validate_cpu.sh
