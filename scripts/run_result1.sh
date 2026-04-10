#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_result1.sh --gpus 1[,5] --experiment EXPERIMENT --model MODEL --seed SEED --name RUN_NAME [-- extra_override=value ...]

Examples:
  scripts/run_result1.sh --gpus 1 --experiment liq_ssm/listops_rs --model soma --seed 1111 --name r1_listops_soma_seed1111
  scripts/run_result1.sh --gpus 1,5 --experiment liq_ssm/pathfinder_rs --model ssm_causal --seed 2222 --name r1_pathfinder_ssm_causal_seed2222 -- data.loader.num_workers=4
EOF
}

gpus=""
experiment=""
model=""
seed=""
run_name=""
python_bin="${PYTHON_BIN:-python}"
declare -a extra_overrides=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      gpus="$2"
      shift 2
      ;;
    --experiment)
      experiment="$2"
      shift 2
      ;;
    --model)
      model="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --name)
      run_name="$2"
      shift 2
      ;;
    --python)
      python_bin="$2"
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

if [[ -z "$gpus" || -z "$experiment" || -z "$model" || -z "$seed" || -z "$run_name" ]]; then
  usage >&2
  exit 1
fi

IFS=',' read -r -a gpu_array <<< "$gpus"
visible_gpu_count="${#gpu_array[@]}"

if [[ "$visible_gpu_count" -eq 1 ]]; then
  trainer_devices='trainer.devices=[0]'
else
  visible_ids=()
  for i in "${!gpu_array[@]}"; do
    visible_ids+=("$i")
  done
  trainer_devices="trainer.devices=[$(IFS=,; echo "${visible_ids[*]}")]"
fi

cmd=(
  "$python_bin" train.py
  "experiment=$experiment"
  "model=$model"
  "run.name=$run_name"
  "run.seed=$seed"
  "$trainer_devices"
)

if [[ "${#extra_overrides[@]}" -gt 0 ]]; then
  cmd+=("${extra_overrides[@]}")
fi

echo "CUDA_VISIBLE_DEVICES=$gpus ${cmd[*]}"
CUDA_VISIBLE_DEVICES="$gpus" "${cmd[@]}"
