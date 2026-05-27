from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_latent_diagnostics(z: torch.Tensor) -> Dict[str, float]:
    """
    Aggregate statistics of the latent batch z: [B, latent_dim].

    Useful for monitoring whether q(z) is converging toward N(0, I):
      - per-dim mean should be near 0
      - per-dim variance should be near 1
      - off-diagonal covariance should be near 0 (factorized)
      - skewness should be near 0, excess kurtosis near 0

    NOTE: interpretability of individual dimensions is NOT guaranteed
    by these metrics. Use probing or intervention experiments separately.
    """
    B, D = z.shape
    eps = 1e-6

    per_dim_mean = z.mean(dim=0)                       # [D]
    per_dim_var = z.var(dim=0, unbiased=False)         # [D]

    mean_abs_mean = per_dim_mean.abs().mean().item()
    var_err_from_1 = (per_dim_var - 1).abs().mean().item()

    # Off-diagonal covariance (subset of dims to keep O(n_sub^2) manageable)
    n_sub = min(128, D)
    z_sub = z[:, :n_sub]
    z_sub_c = z_sub - z_sub.mean(dim=0)
    cov = (z_sub_c.T @ z_sub_c) / max(B - 1, 1)
    off_mask = ~torch.eye(n_sub, dtype=torch.bool, device=z.device)
    off_diag = cov[off_mask].abs()

    # Higher-order marginal moments
    z_c = z - per_dim_mean.unsqueeze(0)
    std = (per_dim_var + eps).sqrt()
    skew = z_c.pow(3).mean(dim=0) / (std.pow(3) + eps)
    kurt = z_c.pow(4).mean(dim=0) / (std.pow(4) + eps)   # raw kurtosis

    return {
        "mean_abs_mean": mean_abs_mean,
        "var_err_from_1": var_err_from_1,
        "off_diag_cov_mean": off_diag.mean().item(),
        "off_diag_cov_max": off_diag.max().item(),
        "skewness_abs_mean": skew.abs().mean().item(),
        "excess_kurtosis_abs_mean": (kurt - 3).abs().mean().item(),
    }


@torch.no_grad()
def compute_z_frechet_distance(z: torch.Tensor, n_proj: int = 512) -> Dict[str, float]:
    """
    Distance of q(z) from N(0, I) via two complementary scalars:

    z_fd_diag  — diagonal Frechet distance (exact when q(z) is factorised):
                 = ||mean_z||_2^2 + sum_i (std_i - 1)^2
                 Interpretable per-dim decomposition; very fast.

    z_fd_proj  — sliced Frechet distance via random unit projections:
                 Each projection of N(0,I) onto a unit vector is N(0,1),
                 so FD per slice = mu_proj^2 + (std_proj - 1)^2.
                 Captures off-diagonal covariance errors that z_fd_diag misses.

    Both decrease monotonically as q(z) → N(0, I).
    z_fd_diag is the more direct ΔFID proxy for well-regularised models.
    """
    B, D = z.shape
    z_f = z.float()

    per_dim_mean = z_f.mean(0)                                  # [D]
    per_dim_std  = z_f.var(0, unbiased=True).clamp(min=0).sqrt()  # [D]

    z_fd_diag = (per_dim_mean.pow(2).sum() + (per_dim_std - 1).pow(2).sum()).item()

    # Random projections: columns of A are unit vectors
    A = F.normalize(torch.randn(D, n_proj, device=z.device, dtype=torch.float32), dim=0)
    y = z_f @ A                                                  # [B, n_proj]
    proj_mean = y.mean(0)
    proj_std  = y.var(0, unbiased=True).clamp(min=0).sqrt()

    z_fd_proj = (proj_mean.pow(2) + (proj_std - 1).pow(2)).mean().item()

    return {"z_fd_diag": z_fd_diag, "z_fd_proj": z_fd_proj}


@torch.no_grad()
def compute_kl_active_dims(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    thresholds: Tuple[float, ...] = (0.1, 0.5, 1.0),
) -> Dict[str, float]:
    """
    Per-dimension KL decomposition and active-unit counts.

    kl_dim_i = 0.5 * (mu_i^2 + exp(logvar_i) - logvar_i - 1)

    Returns kl_total, kl_dim_mean/max/min, and active_dims_at_T for each threshold T.
    Tag format for threshold 0.1 → 'active_dims_at_0_1'.
    """
    kl_dim = 0.5 * (
        mu.float().pow(2) + logvar.float().exp() - logvar.float() - 1
    ).mean(0)  # [D]

    result: Dict[str, float] = {
        "kl_total":    kl_dim.sum().item(),
        "kl_dim_mean": kl_dim.mean().item(),
        "kl_dim_max":  kl_dim.max().item(),
        "kl_dim_min":  kl_dim.min().item(),
    }
    for thr in thresholds:
        tag = f"{thr:g}".replace("-", "m").replace(".", "_")
        result[f"active_dims_at_{tag}"] = float((kl_dim > thr).sum().item())
    return result


@torch.no_grad()
def compute_selective_activity(
    feat_freq: torch.Tensor,
    img_active_counts: torch.Tensor,
    threshold: float,
    latent_dim: int,
) -> Dict[str, float]:
    """
    Feature selectivity metrics derived from per-image KL activity.

    feat_freq         : [D]  fraction of validation images in which each feature
                             had per-image KL > threshold.  Precomputed outside.
    img_active_counts : [N]  number of active features per validation image.
    latent_dim        : D    (needed so img_active_frac is meaningful)

    Desired properties of a healthy overcomplete VAE:
      feat_never_active  → 0    (no dead features)
      feat_always_active → 0    (no always-on / non-selective features)
      feat_freq_p50      → low  (typical feature fires in a minority of images)
      img_active_frac    → low  (each image activates few features — sparse code)
    """
    tag = f"{threshold:g}".replace(".", "_").replace("-", "m")

    # per-feature: activation frequency distribution
    never  = (feat_freq == 0).float().mean().item()
    always = (feat_freq == 1).float().mean().item()
    freq_p10 = feat_freq.quantile(0.10).item()
    freq_p50 = feat_freq.quantile(0.50).item()
    freq_p90 = feat_freq.quantile(0.90).item()
    freq_mean = feat_freq.mean().item()

    # per-image: sparsity (how many features active per image)
    img_counts_f = img_active_counts.float()
    img_mean = img_counts_f.mean().item()
    img_p50  = img_counts_f.quantile(0.50).item()
    img_frac = img_mean / max(latent_dim, 1)

    return {
        f"sel/feat_never_active_{tag}":  never,
        f"sel/feat_always_active_{tag}": always,
        f"sel/feat_freq_mean_{tag}":     freq_mean,
        f"sel/feat_freq_p10_{tag}":      freq_p10,
        f"sel/feat_freq_p50_{tag}":      freq_p50,
        f"sel/feat_freq_p90_{tag}":      freq_p90,
        f"sel/img_active_mean_{tag}":    img_mean,
        f"sel/img_active_p50_{tag}":     img_p50,
        f"sel/img_active_frac_{tag}":    img_frac,
    }


@torch.no_grad()
def compute_marginal_moments(z: torch.Tensor) -> Dict[str, float]:
    """
    Marginal moment statistics of q(z).

    For a well-trained model matching N(0, I):
      mean_abs_mean  → 0
      std_mean       → 1   (std_std → 0 means uniform across dims)
      skew_abs_mean  → 0
      kurt_abs_mean  → 0   (excess kurtosis)
    """
    eps = 1e-6
    z_f = z.float()
    per_dim_mean = z_f.mean(0)
    per_dim_var  = z_f.var(0, unbiased=False)
    std = (per_dim_var + eps).sqrt()

    z_c = z_f - per_dim_mean.unsqueeze(0)
    skew = z_c.pow(3).mean(0) / (std.pow(3) + eps)
    kurt = z_c.pow(4).mean(0) / (std.pow(4) + eps) - 3.0  # excess

    return {
        "marginal_mean_abs_mean": per_dim_mean.abs().mean().item(),
        "marginal_mean_abs_max":  per_dim_mean.abs().max().item(),
        "marginal_std_mean":      std.mean().item(),
        "marginal_std_std":       std.std().item(),
        "marginal_skew_abs_mean": skew.abs().mean().item(),
        "marginal_excess_kurt_abs_mean": kurt.abs().mean().item(),
    }
