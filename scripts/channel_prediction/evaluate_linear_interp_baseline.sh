#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-/home/berkay/Desktop/research/datasets/CSIGen/TemporalWiMAE/ood_test_35/la_1}"
SAVE_DIR="${SAVE_DIR:-results/channel_prediction/linear_interp}"
PILOT_PATTERN="${PILOT_PATTERN:-t:2,11;f:0,2,4,6}"
FREQ_PATCH_SIZE="${FREQ_PATCH_SIZE:-4}"
SNRS="${SNRS:-0,5,10,15,20,25,30}"
SEED="${SEED:-42}"
FREQUENCY_OUTSIDE_MODE="${FREQUENCY_OUTSIDE_MODE:-hold}"
TIME_OUTSIDE_MODE="${TIME_OUTSIDE_MODE:-hold}"

python -m baselines.channel_prediction.evaluate_linear_interp_baseline \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --pilot_pattern "${PILOT_PATTERN}" \
  --freq_patch_size "${FREQ_PATCH_SIZE}" \
  --snrs "${SNRS}" \
  --seed "${SEED}" \
  --frequency_outside_mode "${FREQUENCY_OUTSIDE_MODE}" \
  --time_outside_mode "${TIME_OUTSIDE_MODE}" \
  --output_stem result
