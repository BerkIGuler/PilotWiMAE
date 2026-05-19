#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-/home/berkay/Desktop/research/datasets/CSIGen/TemporalWiMAE/ood_test_35/la_1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/berkay/Desktop/research/2026/PilotWiMAE/runs/decoder_ablations/pilotwimae_dec_only_fst_tk4_sm075_dec12_from_tk2_sm09_fst_scaleaux_noiserobust_snr40/best_checkpoint.pt}"
SAVE_DIR="${SAVE_DIR:-results/channel_prediction/dec12}"

DEVICE="${DEVICE:-cuda:0}"
TEST_SPLIT="${TEST_SPLIT:-0.1}"
NFOLDS="${NFOLDS:-10}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"

PILOT_PATTERN="${PILOT_PATTERN:-t:2,11;f:0,2,4,6}"
SNRS="${SNRS:-0,5,10,15,20,25,30}"

python -m pilotwimae.downstream.channel_prediction.evaluate_mae \
  --data_dir "${DATA_DIR}" \
  --test_split "${TEST_SPLIT}" \
  --n_folds "${NFOLDS}" \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --device "${DEVICE}" \
  --pilot_pattern "${PILOT_PATTERN}" \
  --snrs "${SNRS}" \
  --save_dir "${SAVE_DIR}" \
  --noise_floor
