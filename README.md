# PilotWiMAE

**PilotWiMAE** is a self-supervised framework for learning wireless channel representations directly from sparse, noisy pilot observations. The encoder ingests pilot measurements (not full CSI), with attention factorized along temporal and joint space–frequency axes. Pretraining combines patch-normalized reconstruction (small-scale fading) with an auxiliary scale loss (large-scale fading) and an AWGN curriculum matched to pilot SNR at deployment.

A **decoder-centric second stage** (Phase 2) retrains only the decoder after joint encoder–decoder pretraining, improving channel estimation without degrading representation quality for downstream tasks (beam selection, channel characterization).

## Requirements

### Core (training & evaluation)

| Package | Version | Used for |
|---------|-----------------|----------|
| [PyTorch](https://pytorch.org/) | 2.5.1+cu121 | Models, trainers, KNN / channel-estimation eval |
| [NumPy](https://numpy.org/) | 2.1.2 | Datasets, baselines (linear interp., LMMSE) |
| [PyYAML](https://pyyaml.org/) | 6.0.2 | Config loading |
| [tqdm](https://github.com/tqdm/tqdm) | 4.67.1 | Training and evaluation progress |

### Optional

| Package | Version | Used for |
|---------|-----------------|----------|
| [TensorBoard](https://www.tensorflow.org/tensorboard) | 2.19.0 | Training logs (`logging.tensorboard: true` in configs) |
| [matplotlib](https://matplotlib.org/) | 3.10.0 | Paper figures under `pilotwimae/plot/` |
| [pytest](https://pytest.org/) | 8.4.1 | Unit tests (`tests/`) |


Run from the repository root so `pilotwimae` and `scripts` import correctly (no separate package install required).

## Training

All training commands below assume you run them from the repository root. Replace `cuda:0` with your device as needed. Update `data_dir` paths in the YAML configs before training.

### Phase 1: Joint encoder–decoder pretraining

Full PilotWiMAE pretraining (factorized encoder, scale auxiliary loss, noise-robust SNR curriculum):

```bash
python3 -m scripts.train.train_masked \
  --config configs/train/fst/tk2_sm09_dec2_snr_scale.yaml \
  --device cuda:0
```

### Phase 2: Decoder-only pretraining

Loads the Phase 1 encoder checkpoint and retrains the decoder with a lighter mask (e.g. `tk4_sm075`). Encoder weights are frozen. Decoder is reinitialized.

```bash
python3 -m scripts.train.train_masked \
  --config configs/train/decoder_only/dec_only_fst_tk2_sm09_tk4_sm075_dec2_normpatch_mse.yaml \
  --device cuda:0
```

### Phase 1 ablations

Ablations isolate architectural and objective choices (Phase 1 only, no Phase 2).

| Variant | Description | Command |
|--------|-------------|---------|
| **FST** | Factorized space–time encoder | `python3 -m scripts.train.train_masked --config configs/train/fst/tk2_sm09_dec2.yaml --device cuda:0` |
| **FST scale** | FST + auxiliary scale loss | `python3 -m scripts.train.train_masked --config configs/train/fst/tk2_sm09_dec2_scale.yaml --device cuda:0` |
| **FST noise** | FST + AWGN curriculum on pilots | `python3 -m scripts.train.train_masked --config configs/train/fst/tk2_sm09_dec2_snr.yaml --device cuda:0` |
| **JST** | Joint space–time encoder (no factorization) | `python3 -m scripts.train.train_masked --config configs/train/jst/rm_095_dec2.yaml --device cuda:0` |

The full model (**FST_noise_scale**) combines factorized attention, scale auxiliary loss, and the noise curriculum (Phase 1 config above).

### Supervised baselines

End-to-end supervised models on the same pilot observations, for comparison on downstream tasks:

**Channel estimation**

```bash
python3 -m scripts.train.train_ce \
  --config configs/train/ce/fst_dec2_supervised.yaml \
  --device cuda:0
```

**Beam selection**

```bash
python3 -m scripts.train.train_beam \
  --config configs/train/beam/fst_dec2_o2o2_supervised.yaml \
  --device cuda:0
```

**Channel characterization (LoS / NLoS)**

```bash
python3 -m scripts.train.train_los \
  --config configs/train/los/fst_dec2_supervised.yaml \
  --device cuda:0
```

## Checkpoints (`runs/`)

Training writes checkpoints and resolved configs under `runs/`. Typical layout:

| Directory | Contents |
|-----------|----------|
| `runs/self_supervised/` | PilotWiMAE Phase 1 ablations and full model |
| `runs/decoder_ablations/` | Phase 2 decoder-only runs (various decoder depths) |
| `runs/supervised/` | Supervised baselines (CE, beam, LoS) |

**Self-supervised (Phase 1)**

- `runs/self_supervised/FST/`
- `runs/self_supervised/FST_scale/`
- `runs/self_supervised/FST_noise/`
- `runs/self_supervised/FST_noise_scale/` — full PilotWiMAE (Phase 1)
- `runs/self_supervised/JST/`

**Decoder-only (Phase 2)** — under `runs/decoder_ablations/` (e.g. `pilotwimae_dec_only_fst_tk4_sm075_dec*_from_tk2_sm09_fst_scaleaux_noiserobust_snr40/`).

**Supervised** — under `runs/supervised/` (channel estimation, beam selection, channel characterization).

Each run folder contains `config.yaml` and `best_checkpoint.pt` (or equivalent).

## Evaluation

Set `CHECKPOINT_PATH` (and `DATA_DIR` if needed) in the shell scripts, then run from the repo root.

| Task | Script |
|------|--------|
| Beam selection | `scripts/beam_prediction/evaluate_knn_beam.sh` |
| Channel characterization | `scripts/los/evaluate_knn_los.sh` |
| Channel estimation (learned checkpoint) | `scripts/channel_prediction/evaluate_channel_mae.sh` |
| Channel estimation (linear interpolation baseline) | `scripts/channel_prediction/evaluate_linear_interp_baseline.sh` |
| Channel estimation (LMMSE baseline) | `scripts/channel_prediction/evaluate_lmmse_baseline.sh` |

Example (channel estimation):

```bash
CHECKPOINT_PATH=runs/decoder_ablations/pilotwimae_dec_only_fst_tk4_sm075_dec2_from_tk2_sm09_fst_scaleaux_noiserobust_snr40/best_checkpoint.pt \
  bash scripts/channel_prediction/evaluate_channel_mae.sh
```

KNN evaluation scripts for beam and LoS use frozen encoder representations. Channel estimation evaluation uses the full model (encoder + decoder) for reconstruction.

## License

This project is released under the [MIT License](LICENSE). You may use, copy, modify, and distribute the code and checkpoints for research or commercial purposes with minimal restrictions. Retain the copyright notice in redistributions.
