#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE="${1:-vla}"
PROFILE="${RUN_PROFILE:-diagnostic}"
if [ "$PROFILE" = paper_disposal_proxy ]; then
  if [ "$MODE" != vla ] && [ "$MODE" != baseline ]; then echo 'Paper profile permits VLA or standing diagnostics only' >&2; exit 1; fi
  export TASK_SCENE=paper-disposal CAMERA_PROFILE=approximate-fisheye TOOL_PROFILE=visual-fit
  export OPENHLM_ARM_REFERENCE_SERVO=0 ARM_BIAS_COMPENSATION=0 FUTURE_WINDOW=0
  export OPENHLM_SOURCE_PREVIEW=1 OPENHLM_PREVIEW_SECONDS=.2 EXECUTION_MODE=synchronous TRANSPORT_HZ=30
  export STANDARDIZE_START=1 STANDARDIZE_START_X=-.65
fi
SECONDS_TO_RUN="${EVALUATION_SECONDS:-20}"
RUN="$ROOT/logs/${RUN_NAME:-g1_sim_${MODE}}"
if [ -e "$RUN" ]; then
  echo "Evaluation directory already exists: $RUN" >&2; exit 1
fi
mkdir -p "$RUN"
cd "$ROOT"
export OPENHLM_DDS_DOMAIN="${OPENHLM_DDS_DOMAIN:-37}"
cleanup() {
  touch "$RUN/stop"
  [ -z "${SIM_PID:-}" ] || wait "$SIM_PID" || true
  [ -z "${CONTROL_PID:-}" ] || wait "$CONTROL_PID" || true
}
trap cleanup EXIT
SIM_EXTRA=()
SIM_EXTRA+=(--layout-id "${LAYOUT_ID:-0}")
if [ -n "${WRIST_CALIBRATION_FILE:-}" ]; then SIM_EXTRA+=(--wrist-calibration "$WRIST_CALIBRATION_FILE"); fi
PYTHONPATH="$ROOT/src/GR00T-WholeBodyControl4OpenHLM" \
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${SIM_GPU:-3}" \
  .venv_sim/bin/python -u reproduction/pi05/headless_g1.py \
  --run-dir "$RUN" --duration "$((SECONDS_TO_RUN + 8))" --task "${TASK_SCENE:-none}" --camera-profile "${CAMERA_PROFILE:-pinhole}" --tool-profile "${TOOL_PROFILE:-legacy}" --run-label "$MODE" "${SIM_EXTRA[@]}" > "$RUN/simulator.log" 2>&1 &
SIM_PID=$!
for _ in $(seq 1 120); do
  [ -f "$RUN/sim-ready" ] && break
  kill -0 "$SIM_PID" || exit 1
  sleep 1
done
CUDA_VISIBLE_DEVICES="${CONTROL_GPU:-4}" .venv/bin/python -u reproduction/pi05/run_controller.py \
  --run-dir "$RUN" > "$RUN/controller-runner.log" 2>&1 &
CONTROL_PID=$!
CLIENT_EXTRA=()
if [ "$PROFILE" = paper_disposal_proxy ]; then CLIENT_EXTRA+=(--initial-pose raised --warmup-queries 10); fi
if [ "${FUTURE_WINDOW:-0}" = 1 ]; then CLIENT_EXTRA+=(--future-window); fi
if [ "${ARM_BIAS_COMPENSATION:-0}" = 1 ]; then CLIENT_EXTRA+=(--arm-bias-compensation); fi
if [ "${STANDARDIZE_START:-0}" = 1 ]; then CLIENT_EXTRA+=(--standardize-start); fi
CLIENT_EXTRA+=(--start-x "${STANDARDIZE_START_X:-0}")
if [ -n "${REPLAY_RUN:-}" ]; then CLIENT_EXTRA+=(--replay-run "$REPLAY_RUN"); fi
.venv/bin/python -u reproduction/pi05/evaluate_g1_client.py \
  --run-dir "$RUN" --mode "$MODE" --seconds "$SECONDS_TO_RUN" --execution "${EXECUTION_MODE:-synchronous}" --transport-hz "${TRANSPORT_HZ:-30}" "${CLIENT_EXTRA[@]}" > "$RUN/client.log" 2>&1
touch "$RUN/stop"
wait "$SIM_PID"
wait "$CONTROL_PID"
cat "$RUN/simulation.json" "$RUN/client.json"
