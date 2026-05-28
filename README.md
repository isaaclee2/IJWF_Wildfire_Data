# Probabilistic Forecasting of Wildfire Size Using Conditional Flow Matching

Code and data for the paper:

> Isaac Lee, Bryan Shaddy, Assad Oberai. *Probabilistic Forecasting of Wildfire Size Using Conditional Flow Matching.* (2026)

## Overview

This repository contains the training data and model code for a probabilistic surrogate model that predicts wildfire area growth using conditional flow matching (CFM). Given the current and previous burned area alongside spatially averaged weather, terrain, and fuel conditions, the model generates an ensemble of fire area growth predictions over a 3-hour interval. Applied autoregressively, it produces 24-hour fire area trajectories with calibrated uncertainty estimates.

## Repository Structure

```
.
├── datasets/
│   ├── wildfire_data.npz               # Training and test data (X_train, Y_train, X_test, Y_test)
│   └── sensitivity_data.npz            # Data for sensitivity analysis
├── 24hr_test_data/
│   ├── WEST_HEMP_test_data.npy         # Per-fire conditioning inputs for 24-hour rollouts
│   ├── TUMTUM_test_data.npy
│   └── ...                             # One .npy file per held-out test fire (12 total)
├── normalization_data/
│   ├── lognorm_x1_log_max.npy          # Log-normalization constants for output denormalization
│   ├── lognorm_x2_log_max.npy
│   └── lognorm_x3_log_max.npy
├── model/
│   ├── flow.py                         # Flow matching model and neural network architectures
│   ├── trainer.py                      # Training and sampling routines
│   ├── data_reader.py                  # Data loading and normalization utilities
│   ├── optimal_transport.py            # Optional OT plan sampler
│   ├── config.yml                      # Hyperparameters and paths
│   └── environment.yml                 # Conda environment specification
└── shell_scripts/
    ├── run_train.sh                    # SLURM job script for training
    └── run_sample.sh                   # SLURM job script for sampling/evaluation
```

## Data

`datasets/wildfire_data.npz` contains the following arrays:

| Key | Shape | Description |
|---|---|---|
| `X_train` | (14000, 24) | Training inputs |
| `Y_train` | (14000, 3) | Training targets (log-normalized fire area increments) |
| `X_test` | (1200, 24) | Test inputs |
| `Y_test` | (1200, 3) | Test targets |

**Input features** (`X`, dim=24): current burned area, previous burned area, zonal wind, meridional wind, relative humidity, temperature, 4 terrain metrics, 14 fuel category fractions.

**Outputs** (`Y`, dim=3): log-normalized change in fire area at 1, 2, and 3 hours beyond the current timestep. Invert with `exp(y * y_max) - 1` to recover acres.

Data are derived from 152 WRF-SFIRE coupled atmosphere–wildfire simulations of 2023 CONUS wildfire events. 140 simulations are used for training and 12 are held out for testing. Each simulation is augmented with 10 random rotations and 10 forecast time samples, yielding 14,000 training samples and 1,200 test samples. See the paper (Section 2.4) for full details on normalization and data augmentation.

`24hr_test_data/` contains the per-timestep conditioning inputs for each of the 12 held-out test fires, used for the autoregressive 24-hour rollouts in Section 3.3. `normalization_data/` contains the three log-normalization constants needed to convert model outputs back to acres during inference.

## Setup

```bash
conda env create -f model/environment.yml
conda activate env_red
```

Key dependencies: Python 3.10, PyTorch 2.4.0, CUDA 11.8, SciPy, NumPy, `ema-pytorch`.

## Configuration

Edit `model/config.yml` before running. At minimum, update the data paths:

```yaml
data:
  data_path: "/path/to/datasets/wildfire_data.npz"
  save_path_normalization_scale_mean: "/path/to/datasets"
```

Key hyperparameters (defaults match the paper):

| Parameter | Value | Description |
|---|---|---|
| `model.width` | 256 | Hidden layer width |
| `model.depth` | 4 | Number of hidden layers |
| `model.network_type` | `modelNN2` | Architecture with Fourier time embedding |
| `training.n_iters` | 20000 | Number of mini-batches |
| `training.batch_size` | 2000 | Batch size |
| `training.lr` | 1e-3 | Adam learning rate |

## Training

```bash
cd model
python main.py --config config.yml --save_dir run_wildfire --train --ni
```

The `--ni` flag clears and recreates the log directory for a fresh run (omit to resume into an existing directory). Logs and checkpoints are written to `model/exp/logs/Gaussian/run_wildfire/`. EMA weights are saved as `ema_<iter>.pt` every `save_freq` iterations; the final checkpoint is `ema_0.pt`.

Or submit to a SLURM cluster:

```bash
sbatch shell_scripts/run_train.sh
```

## Sampling

```bash
cd model
python main.py --config config.yml --save_dir run_wildfire --sample --checkpoint <iteration>
```

Set `--checkpoint 0` to load the final checkpoint. Output is written to `model/exp/sampling/Gaussian/run_wildfire/ch_<iteration>/`. Generated single-step samples are saved as `generated_samples.pt`; autoregressive 24-hour rollout results are saved as `*_fire_area_matrix.npz`.

Or submit to SLURM:

```bash
sbatch shell_scripts/run_sample.sh
```

## Model Details

The velocity field is parameterized by a fully connected feed-forward network (`modelNN2`) with 4 hidden layers of width 256 and a 4-component Fourier time embedding. The model is trained with the CFM objective (Equation 3 in the paper) using the Adam optimizer and an exponential moving average (EMA, decay=0.9999) of model parameters. The EMA weights are used at inference time.

Output fire area increments are log-normalized prior to training (Equation 4 in the paper), which enforces non-negativity and compresses the heavy-tailed distribution of fire growth.

## Citation

```bibtex
@article{lee2026wildfire,
  title={Probabilistic Forecasting of Wildfire Size Using Conditional Flow Matching},
  author={Lee, Isaac and Shaddy, Bryan and Oberai, Assad},
  year={2026}
}
```
