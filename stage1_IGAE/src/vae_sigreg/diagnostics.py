from typing import Dict

import torch


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
