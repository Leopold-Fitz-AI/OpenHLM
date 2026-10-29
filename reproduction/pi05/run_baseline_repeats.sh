#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export EVALUATION_SECONDS=360 STANDARDIZE_START=1 STANDARDIZE_START_X=.05 FUTURE_WINDOW=1 CONTROL_GPU=0
export TASK_SCENE=bottle-bin TOOL_PROFILE=visual-fit CAMERA_PROFILE=approximate-fisheye OPENHLM_PREVIEW_SECONDS=.2 OPENHLM_ARM_REFERENCE_SERVO=1
.venv_sim/bin/python reproduction/pi05/check_baseline_success.py logs/g1_script_servo07
for index in 08 09 10 11; do
  export RUN_NAME="g1_script_servo$index"
  printf '%s\n' "$RUN_NAME" > logs/scripted_suite_current.txt
  bash reproduction/pi05/run_sim_evaluation.sh scripted-baseline > "logs/$RUN_NAME.runner.log" 2>&1
  .venv_sim/bin/python reproduction/pi05/check_baseline_success.py "logs/$RUN_NAME"
done
.venv_sim/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
trials=[json.loads(Path(f'logs/g1_script_servo{i:02}/baseline_gate.json').read_text()) for i in range(7,12)]
paths=['reproduction/pi05/g1_scripted_baseline.py','reproduction/pi05/headless_g1.py','reproduction/pi05/g1_render_worker.py','reproduction/pi05/evaluate_g1_client.py','reproduction/pi05/preview_stream.py','src/GR00T-WholeBodyControl4OpenHLM/gear_sonic_deploy/target/release/g1_deploy_onnx_ref']
s={'protocol':'G1 scripted pick-and-bin / original SONIC balance plus right-arm reference PI / fixed working x=.05m','not_vla':True,'five_consecutive_passed':all(t['passed'] for t in trials),'trials':trials,'current_source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
Path('logs/scripted_baseline_suite.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
PY
