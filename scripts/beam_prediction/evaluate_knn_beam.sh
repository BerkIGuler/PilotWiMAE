#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="/home/berkay/Desktop/research/datasets/CSIGen/TemporalWiMAE/ood_test_28/la_1"
CHECKPOINT_PATH="runs/self_supervised/JST/best_checkpoint.pt"
SAVE_DIR="results/beam_prediction/"

DEVICE="cuda:0"
TEST_SPLIT=0.1
NFOLDS=10
SEED=42
BATCH_SIZE=512
NUM_WORKERS=4

POOLING="mean"
INFERENCE_TOKEN_MODE="${INFERENCE_TOKEN_MODE:-pilot_visible}"  # full_grid | masked_visible | pilot_visible
PILOT_PATTERN="${PILOT_PATTERN:-t:2,11;f:0,2,4,6}"

N_H=8   # horizontal UPA elements
N_V=4   # vertical UPA elements
O_H=2   # horizontal oversampling factor
O_V=2   # vertical oversampling factor
ANTENNA_ORDER="hv"  # or "vh"
K=20
METRIC="cosine"  # or "euclidean"

# Comma-separated SNRs (dB)
SNRS="0,5,10,15,20,25,30"

EVAL_EXTRA=()
if [[ -n "${PILOT_PATTERN}" ]]; then
  EVAL_EXTRA+=(--pilot_pattern "${PILOT_PATTERN}")
fi  

python -m pilotwimae.downstream.beam_prediction.evaluate_knn \
  --data_dir "${DATA_DIR}" \
  --test_split "${TEST_SPLIT}" \
  --n_folds "${NFOLDS}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --device "${DEVICE}" \
  --pooling "${POOLING}" \
  --inference_token_mode "${INFERENCE_TOKEN_MODE}" \
  --n_h "${N_H}" \
  --n_v "${N_V}" \
  --o_h "${O_H}" \
  --o_v "${O_V}" \
  --antenna_order "${ANTENNA_ORDER}" \
  --k "${K}" \
  --metric "${METRIC}" \
  --snrs "${SNRS}" \
  --save_dir "${SAVE_DIR}" \
  --output_stem "result_28GHz_FST_noise_scale" \
  "${EVAL_EXTRA[@]}"