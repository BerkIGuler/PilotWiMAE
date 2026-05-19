set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-/home/berkay/Desktop/research/datasets/CSIGen/TemporalWiMAE/ood_test_35}"
TEST_DIR="${TEST_DIR:-/home/berkay/Desktop/research/datasets/CSIGen/TemporalWiMAE/ood_test_35/la_1}"

CORR_DIRS="${CORR_DIRS:-${DATA_DIR}}"
SAVE_DIR="${SAVE_DIR:-results/channel_prediction/lmmse/}"
PILOT_PATTERN="${PILOT_PATTERN:-t:2,11;f:0,2,4,6}"
FREQ_PATCH_SIZE="${FREQ_PATCH_SIZE:-4}"
SNRS="${SNRS:-0,5,10,15,20,25,30}"
SEED="${SEED:-42}"
NUM_FOLDS="${NUM_FOLDS:-10}"
DEBUG_SIZE="${DEBUG_SIZE:-0}"
DEBUG_CORR_SIZE="${DEBUG_CORR_SIZE:-0}"
OUTPUT_STEM="${OUTPUT_STEM:-results_kronecker_matchedR}"

PILOT_SNR_MODE="${PILOT_SNR_MODE:-per_channel}"
LMMSE_MODEL="${LMMSE_MODEL:-kronecker}"
SOLVER_BACKEND="${SOLVER_BACKEND:-torch}"
TORCH_DEVICE="${TORCH_DEVICE:-cuda:0}"
TORCH_BATCH_SIZE="${TORCH_BATCH_SIZE:-64}"

read -r -a CORR_ARR <<< "${CORR_DIRS}"

python -m baselines.channel_prediction.evaluate_lmmse_baseline \
  --corr_dirs "${CORR_ARR[@]}" \
  --test_dir "${TEST_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --pilot_pattern "${PILOT_PATTERN}" \
  --freq_patch_size "${FREQ_PATCH_SIZE}" \
  --snrs "${SNRS}" \
  --seed "${SEED}" \
  --num_folds "${NUM_FOLDS}" \
  --debug_size "${DEBUG_SIZE}" \
  --debug_corr_size "${DEBUG_CORR_SIZE}" \
  --pilot_snr_mode "${PILOT_SNR_MODE}" \
  --lmmse_model "${LMMSE_MODEL}" \
  --solver_backend "${SOLVER_BACKEND}" \
  --torch_device "${TORCH_DEVICE}" \
  --torch_batch_size "${TORCH_BATCH_SIZE}" \
  --output_stem "${OUTPUT_STEM}"
