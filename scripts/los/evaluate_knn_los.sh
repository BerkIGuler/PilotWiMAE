set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="${DATA_DIR:-/home/berkay/Desktop/research/datasets/CSIGen/TemporalWiMAE/ood_test_28/la_1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-runs/self_supervised/JST/best_checkpoint.pt}"
SAVE_DIR="${SAVE_DIR:-results/los/}"

DEVICE="${DEVICE:-cuda:0}"
TEST_SPLIT="${TEST_SPLIT:-0.1}"
NFOLDS="${NFOLDS:-10}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-4}"

POOLING="${POOLING:-mean}"
INFERENCE_TOKEN_MODE="${INFERENCE_TOKEN_MODE:-pilot_visible}"  # full_grid | pilot_visible
PILOT_PATTERN="${PILOT_PATTERN:-t:2,11;f:0,2,4,6}"

K="${K:-20}"
METRIC="${METRIC:-cosine}"
SNRS="${SNRS:-0,5,10,15,20,25,30}"

EXTRA=()
if [[ -n "${PILOT_PATTERN:-}" ]]; then
  EXTRA+=(--pilot_pattern "${PILOT_PATTERN}")
fi

python -m pilotwimae.downstream.los.evaluate_knn \
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
  --k "${K}" \
  --metric "${METRIC}" \
  --snrs "${SNRS}" \
  --save_dir "${SAVE_DIR}" \
  "${EXTRA[@]}"
