# IGAE Stage-1

Overcomplete VAE on DINO features (FFHQ-256).

## Kept Experiments

The retained trained artifacts are the default Stage-1 grid:

- loss: sample KL + D-space Epps-Pulley SIGReg
- checkpoint prefix: `s1_grid_*`
- decoder default: MLP decoder; linear decoder variants are kept where present
- data: DINO ViT-B/8 features, FFHQ-256
- latent_dim: 6144
- default physical batch size: 2048

Artifacts kept:

- `checkpoints/s1_grid*/`
- grid-related visualizations under `visualizations/`
- grid-related logs under `logs/`

## Current Sigma-Confidence Probes

These are active exploratory runs on top of the default grid setting
`beta_kl=1e-4`, `lambda_sigreg=100`, MLP decoder, batch size 2048.
The motivation is to make `sigma` act as image/feature confidence while keeping
feature usage distributed instead of concentrated in a few global channels.

### Sigma Mean Deviation

Initial regularizer:

```text
sigma_mean_d = mean_B(sigma[:, d])
loss = mean_d |target - sigma_mean_d| or (target - sigma_mean_d)^2
```

with `only_below=True`, so only feature means below the target are penalized.

Runs:

- `s1_grid_b1e-4_l100_sigmadev_l1_sd0p1_below`
- `s1_grid_b1e-4_l100_sigmadev_l2_sd0p1_below`
- `s1_grid_b1e-4_l100_sigmadev_l1_sd0p5_below`
- `s1_grid_b1e-4_l100_sigmadev_l2_sd0p5_below`
- `s1_grid_b1e-4_l100_sigmadev_l1_sd0p05_t0p5_below`

Observations:

- `target=1.0` with stronger weights made sigma means behave, but reduced useful
  latent usage and worsened reconstruction.
- `target=0.5` was too weak for the intended behavior; the penalty was nearly
  inactive because feature mean sigma stayed well above 0.5.
- Mean-sigma constraints do not distinguish selective confidence from a feature
  being globally low-sigma.

### Sigma Low-Fraction Upper Bound

Next probe constrained the fraction of images for which a feature is low-sigma:

```text
low_frac_d = mean_B[sigma[:, d] < 0.8]
loss = mean_d relu(low_frac_d - 0.1)
```

Run:

- `s1_grid_b1e-4_l100_siglowfrac_t0p8_f0p1_l0p05`

Observation:

- This suppresses features that are low-sigma for too many images.
- By itself it does not encourage dead features to become selectively confident.

### Soft Sigma Low-Fraction Band

Current best probe:

```text
low_score_bd = sigmoid((0.8 - sigma_bd) / 0.05)
low_frac_d = mean_B(low_score_bd)
loss = mean_d [relu(0.02 - low_frac_d) + relu(low_frac_d - 0.10)]
```

Run:

- `s1_grid_b1e-4_l100_siglowband_t0p8_min0p02_max0p1_l0p05`

End-of-training snapshot:

- `rec ~= 0.1817`
- `kl_total ~= 217.5`
- `active@0.1 = 380`
- `feat_never_active@0.1 = 0.0153`
- `img_active_mean@0.1 = 490.7`
- `soft low_frac mean/p50/p90 = 0.0396 / 0.0310 / 0.0658`
- `hard low_frac mean/p50/p90 = 0.0001 / 0.0000 / 0.0000`
- `frac(features under min) = 0.0000`

Interpretation:

- The soft band dramatically reduced dead features under the KL>0.1 activity
  metric.
- However, it did not create many truly low `sigma < 0.8` or `sigma < 0.9`
  responses. It mainly satisfied the soft sigmoid score.
- If the goal is to have some images per feature reach `sigma ~= 0.5`, the next
  candidate should directly target the low tail, e.g. bottom-2% or bottom-5%
  sigma per feature pulled toward 0.5, plus an upper-bound guard to prevent
  global low-sigma channels.

### KL plus `|mu| * sigma` Coupling

Follow-up probe:

```text
loss = reconstruction + beta_kl * KL + lambda * mean_BD(|mu| * sigma)
```

Motivation:

- If a feature carries information through a large `|mu|`, lowering `sigma`
  reduces the `|mu| * sigma` penalty.
- The hope was that this would couple confidence to active `mu`, while the KL
  term would keep most inactive dimensions near the prior.

Runs:

- `s1_grid_b1e-4_kl_musigl1_l0p05`
- `s1_grid_b1e-4_kl_musigl1_l0p1`
- `s1_grid_b1e-4_kl_musigl1_l0p5`

Snapshots:

| run | status | rec | kl | active@0.1 | notes |
|---|---:|---:|---:|---:|---|
| `lambda=0.05` | done | `~0.103` | `~172` | `97` | many dead features; max-KL dim remains global low-sigma |
| `lambda=0.1` | done | `~0.102` | `~187` | `90` | stronger `mu` shrinkage; max-KL dim `sigma ~= 0.123` |
| `lambda=0.5` | running/early negative | `~0.306` | `~95` | very low | near-collapse; `dead ~= 0.993`, `img_frac ~= 0.007` around epoch 150 |

Interpretation:

- This objective did not solve the global low-sigma channel problem.
- Increasing `lambda` mostly reduced `mu` usage and increased dead features.
- The easy solutions are:
  - important dims keep large `mu` and lower `sigma`, often globally;
  - most other dims reduce `mu` toward zero and become inactive.
- The loss has no term that controls per-feature low-sigma frequency, image-wise
  low-sigma budget, or coverage across features, so it cannot distinguish
  selective confidence from a few global confidence channels.

Current conclusion:

- `|mu| * sigma` is useful as a negative result.
- It couples active `mu` and low `sigma`, but in this setup it behaves more like
  `mu` pruning plus a few global low-sigma channels than like selective
  per-image confidence.

Recommended next direction:

```text
confidence = relu(1 - sigma)
active_gate = stopgrad(sigmoid((|mu| - threshold) / temperature))
```

Use explicit selectivity constraints rather than only a pointwise product:

- penalize low sigma when `mu` is inactive:
  `mean((1 - active_gate) * confidence)`;
- cap per-feature low-sigma frequency:
  `relu(mean_B[sigma < threshold] - max_frac)`;
- cap per-image low-sigma budget:
  `relu(mean_D[sigma < threshold] - max_dims_per_image)`;
- add a coverage term only if dead features must be reduced, e.g. a very soft
  minimum low-sigma frequency or active frequency target, but avoid forcing all
  features to be confident.

## Pruning and Feature Visualizations

Post-hoc pruning:

- Script: `scripts/lat_pruning_eval.py`
- For MLP decoders, supported criteria are `kl`, `l1`, and `mu_var`.
- `dec_norm` is available only for linear-decoder checkpoints.
- The pruning test replaces pruned dims either with prior samples or zeros and
  plots reconstruction MSE versus number of dims kept.

Key pruning observations:

- Baseline `s1_grid_b1e-4_l100` and sigma-regularized variants are saved under
  `visualizations/pruning/`.
- For `s1_grid_b1e-4_l100_sigmadev_l1_sd0p5_below`, KL and L1 pruning behaved
  similarly. The 1% free-prune point was roughly `prune 335 / keep 5809`, so
  prior-fill pruning was not very forgiving despite many features having low
  per-image KL selectivity.

Feature image probes:

- `scripts/query_kl_top_dims_grid.py` selects top-K latent dims by per-image KL
  for a query image and sorts those dims by query `mu`.
- `scripts/feature_topkl_sorted_by_mu.py` selects top-K images by per-image KL
  for one feature and sorts those images by `mu`.
- `scripts/feature_mu_extreme_grid.py` shows the most negative and most positive
  `mu` images for one feature.

Recent feature probes used image index `2685` and the high-KL feature dims:

- `4560`
- `1947`
- `2668`
- `264`
- `1414`
- `3547`

Outputs are under:

- `visualizations/query_kl_top_dims/`
- `visualizations/feature_topkl_mu_sorted/`
- `visualizations/feature_mu_extremes/`

## Pruned Experiment Traces

The following additional Stage-1 directions were tried, then their training
code, checkpoints, logs, and non-grid visualizations were pruned to keep only
the default grid path:

- low-beta sample KL sweeps: `s1_lowb_*`, `s1_vlowb_*`
- tier sweeps for beta/lambda/warmup selection: `s1_tierA_*`, `s1_tierB_*`, `s1_tierC_*`
- feature-level KL variants: `s1_featkl_*`
- squared feature-level KL: `s1_featkl_sq_*`
- log feature-level KL variants: `s1_featkl_log_*`, `s1_featkl_newloss_*`
- sample KL plus L1 on `mu`: `s1_samplel1_*`
- feature-axis SIGReg: `s1_featsr_*`
- SIGReg-only beta-zero runs: `s1_sigreg_only_*`
- B-space SIGReg: `s1_bspace_sigreg_*`
- deterministic AE / no-reparameterization runs: `s1_det_ae_*`

## Main Files

| File | Role |
|---|---|
| `src/vae_sigreg/model.py` | `OvercompleteVariationalAE` architecture |
| `src/vae_sigreg/losses.py` | reconstruction, KL, and SIGReg losses |
| `src/vae_sigreg/diagnostics.py` | latent diagnostics |
| `src/vae_sigreg/sample.py` | prior sampling helper |
| `train_run.py` | Stage-1 training entry point |
