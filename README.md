# Visibility Regimes

Research companion for **Anticipating regime transitions in stationary chaotic
systems using visibility-graph-based structural representations**.

## Install and use

```sh
git clone https://github.com/acandreani/visibility-regimes.git
cd visibility-regimes
python -m pip install -e .
python -m unittest -v
python reproduce.py
```

```python
import numpy as np
from visibility_regimes import augment

observed = np.random.default_rng(1).normal(size=(100, 24, 2))
features = augment(observed)  # shape (100, 28, 2)
```

Only the latest `L=min(W,8)` observed samples form each natural visibility
graph. Four signed node strengths are appended in reverse temporal order.
The function never accepts labels or future observations. Unweighted adjacency
invariance does not imply invariance of these amplitude-dependent weights.

For training, install `pip install -e '.[train]'`, then call
`fit_predict(train_windows, train_labels, test_windows, graph=True, device='cpu')`.
Use `graph=False` for the baseline. CUDA requires a compatible PyTorch install;
set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching GPU training. The research
environment used Python 3.11, NumPy 1.26.4 and PyTorch 2.2.2+cu118.

## Reproduction scope (v0.1.0)

The release verifies **1512 synthetic prediction arrays**, representing 252
evaluable cells of the updated four-system heatmaps (Figure 7). Twelve logistic
cells remain unavailable because not all folds contain both classes. SHA-256
checksums, raw/graph label equality, per-fold and pooled macro F1 are checked.
`output/heatmaps.json` contains the regenerated numerical grids. Figure 7 uses
pooled relative gain for Ikeda and differences of fold means in percentage
points for the other systems; these statistics are NOT interchangeable.

Grid: N=100000, W=4,6,8,10,24,44,64,84,104,124,144; T=1..6; three chronological
folds; 100 epochs; batch size 32; cutoff strictly greater than 0.5. W=4 uses
kernel 2 in both representations; other windows use kernel 3. Four appended
strengths increase model capacity relative to the unpadded raw baseline.

The public training helper follows the exploratory PyTorch protocol for
logistic, Henon and switching laser (seed 20261022, Adam lr=.001, eps=1e-7,
unweighted BCE, no normalization). The released Ikeda predictions instead
come from the recovered Keras 2.14/TensorFlow 2.14.1 implementation, with seeds
20261027+fold. They are not claimed to be reproduced by the PyTorch helper.

**This release reproduces the heatmap metrics from saved predictions, not all
training runs from initial conditions.** Trajectory generation, full campaign
orchestration, the earlier window/length curves, and equal-capacity/parameter
selection tables are not packaged in this release. The article's within-horizon
table protocol differs from the segment-based exploratory sampler provided
here. Do not silently substitute one for the other. GPU results need not be
bitwise identical. The grids are exploratory and reuse previously examined folds.

## Privacy and provenance

This is a fresh, allowlisted export, not a copy of the working repository or
its Git history. Only numerical simulated labels/probabilities and selected
research functions are included. No correspondence, reviewer material, PDFs,
credentials, machine paths, host addresses, logs or infrastructure metadata
are distributed. The library contains the authors' corrected NumPy/PyTorch
implementation, not the recovered legacy source snapshots. No third-party
packages are vendored. This repository is released under the MIT license;
no license for redistribution is implied for material outside this repository.
