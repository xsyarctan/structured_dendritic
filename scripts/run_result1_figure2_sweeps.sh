#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF_USAGE'
Usage:
  scripts/run_result1_figure2_sweeps.sh [--dry-run] [--num-workers N] [-- extra_override=value ...]

What it launches by default:
  Main Figure 2 sweeps:
    listops     -> liq_ssm/listops_rs      on GPU 1
    text/imdb   -> liq_ssm/imdb_rs         on GPU 5
    pathfinder  -> liq_ssm/pathfinder_rs   on GPU 7
    seeds       -> 1111,2222,3333
    models      -> soma, pointwise_control, conv1d_reservoir, ssm_reservoir, ssm_causal

  S4 debug references:
    listops     -> s4_debug/listops        on GPU 0
    text/imdb   -> s4_debug/imdb           on GPU 2
    pathfinder  -> s4_debug/pathfinder     on GPU 4
    seeds       -> 1111

Examples:
  scripts/run_result1_figure2_sweeps.sh
  scripts/run_result1_figure2_sweeps.sh --dry-run
  scripts/run_result1_figure2_sweeps.sh --num-workers 8 -- logging.kind=csv

Optional environment overrides:
  MAIN_SEEDS_CSV=1111,2222,3333
  MAIN_MODELS_CSV=soma,pointwise_control,conv1d_reservoir,ssm_reservoir,ssm_causal
  S4_SEEDS_CSV=1111
  PYTHON_BIN=python
EOF_USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT_DIR/scripts/run_result1.sh"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/launch_logs/result1_figure2_$TIMESTAMP"

dry_run=0
num_workers="${NUM_WORKERS:-4}"
declare -a extra_overrides=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    --num-workers)
      num_workers="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      extra_overrides=("$@")
      break
      ;;
    *)
      extra_overrides+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "$RUNNER" ]]; then
  echo "Runner script not found or not executable: $RUNNER" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a main_seeds <<< "${MAIN_SEEDS_CSV:-1111,2222,3333}"
IFS=',' read -r -a main_models <<< "${MAIN_MODELS_CSV:-soma,pointwise_control,conv1d_reservoir,ssm_reservoir,ssm_causal}"
IFS=',' read -r -a s4_seeds <<< "${S4_SEEDS_CSV:-1111}"

model_alias() {
  case "$1" in
    soma) echo "soma" ;;
    pointwise_control) echo "pointwise" ;;
    conv1d_reservoir) echo "conv_reservoir" ;;
    ssm_reservoir) echo "ssm_reservoir" ;;
    ssm_causal) echo "ssm_causal" ;;
    s4d_standard_bidir) echo "s4_debug" ;;
    *) echo "$1" ;;
  esac
}

run_single() {
  local gpu="$1"
  local experiment="$2"
  local model="$3"
  local seed="$4"
  local run_name="$5"

  local -a cmd=(
    "$RUNNER"
    --gpus "$gpu"
    --experiment "$experiment"
    --model "$model"
    --seed "$seed"
    --name "$run_name"
    --
    "data.loader.num_workers=$num_workers"
  )

  if [[ "${#extra_overrides[@]}" -gt 0 ]]; then
    cmd+=("${extra_overrides[@]}")
  fi

  if [[ "$dry_run" -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}"
}

run_main_task_sweep() {
  local task_tag="$1"
  local experiment="$2"
  local gpu="$3"

  for seed in "${main_seeds[@]}"; do
    for model in "${main_models[@]}"; do
      run_single \
        "$gpu" \
        "$experiment" \
        "$model" \
        "$seed" \
        "r1_${task_tag}_$(model_alias "$model")_seed${seed}"
    done
  done
}

run_s4_debug_task() {
  local task_tag="$1"
  local experiment="$2"
  local gpu="$3"

  for seed in "${s4_seeds[@]}"; do
    run_single \
      "$gpu" \
      "$experiment" \
      "s4d_standard_bidir" \
      "$seed" \
      "r1_${task_tag}_s4_debug_seed${seed}"
  done
}

declare -a child_pids=()
declare -a child_labels=()

launch_worker() {
  local label="$1"
  local log_file="$2"
  shift 2

  (
    echo "[$(date '+%F %T')] starting $label"
    "$@"
    echo "[$(date '+%F %T')] finished $label"
  ) > >(tee -a "$log_file") 2>&1 &

  child_pids+=("$!")
  child_labels+=("$label")
}

cleanup() {
  if [[ "${#child_pids[@]}" -eq 0 ]]; then
    return
  fi
  echo "Stopping child sweeps..." >&2
  for pid in "${child_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}

trap cleanup INT TERM

echo "Launching Result 1 / Figure 2 sweeps"
echo "Logs: $LOG_DIR"
echo "Main seeds: ${main_seeds[*]}"
echo "Main models: ${main_models[*]}"
echo "S4 seeds: ${s4_seeds[*]}"
echo

launch_worker \
  "main_listops_gpu1" \
  "$LOG_DIR/main_listops_gpu1.log" \
  run_main_task_sweep listops liq_ssm/listops_rs 1

launch_worker \
  "main_text_gpu5" \
  "$LOG_DIR/main_text_gpu5.log" \
  run_main_task_sweep text liq_ssm/imdb_rs 5

launch_worker \
  "main_pathfinder_gpu7" \
  "$LOG_DIR/main_pathfinder_gpu7.log" \
  run_main_task_sweep pathfinder liq_ssm/pathfinder_rs 7

launch_worker \
  "s4_listops_gpu0" \
  "$LOG_DIR/s4_listops_gpu0.log" \
  run_s4_debug_task listops s4_debug/listops 0

launch_worker \
  "s4_text_gpu2" \
  "$LOG_DIR/s4_text_gpu2.log" \
  run_s4_debug_task text s4_debug/imdb 2

launch_worker \
  "s4_pathfinder_gpu4" \
  "$LOG_DIR/s4_pathfinder_gpu4.log" \
  run_s4_debug_task pathfinder s4_debug/pathfinder 4

status=0
for i in "${!child_pids[@]}"; do
  if ! wait "${child_pids[$i]}"; then
    echo "Worker failed: ${child_labels[$i]}" >&2
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "One or more sweeps failed. Check logs under $LOG_DIR" >&2
  exit "$status"
fi

echo "All requested sweeps finished successfully."
