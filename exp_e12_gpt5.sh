#!/usr/bin/env bash

set -euo pipefail

# First-round fragment benchmark runs:
# - datasets: active site + binding site
# - model: DeepSeek V3.1 via OpenRouter
# - experiments: E1, E2
# - split: test
# - size: full chosen split
# - parallelism: controlled by MAX_PARALLEL

MODEL_PROVIDER="openrouter"
MODEL_NAME="openai/gpt-5"
SPLIT="test"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

run_benchmark() {
  local dataset_id="$1"
  local experiment="$2"

  echo "[start] dataset=${dataset_id} experiment=${experiment} split=${SPLIT}"
  python -m evaluation_llm \
    --dataset_id "${dataset_id}" \
    --split "${SPLIT}" \
    --experiment "${experiment}" \
    --model_provider "${MODEL_PROVIDER}" \
    --model_name "${MODEL_NAME}"
  echo "[done] dataset=${dataset_id} experiment=${experiment} split=${SPLIT}"
}

launch_run() {
  local dataset_id="$1"
  local experiment="$2"

  run_benchmark "${dataset_id}" "${experiment}" &

  while (( $(jobs -pr | wc -l) >= MAX_PARALLEL )); do
    wait -n
  done
}

launch_run "VenusX_Res_Act_MF50" "E1"
launch_run "VenusX_Res_Act_MF50" "E2"
launch_run "VenusX_Res_BindI_MF50" "E1"
launch_run "VenusX_Res_BindI_MF50" "E2"

wait
