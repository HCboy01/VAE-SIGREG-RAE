from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """MSE between reconstruction and input."""
    return F.mse_loss(x_hat, x)


def kl_bottleneck_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    VAE KL divergence KL(q(z|x) || N(0,I)).

    Limits information capacity per sample: encourages each encoder output
    distribution to stay close to the prior N(0, I).

    KL = 0.5 * mean_batch[ sum_i( mu_i^2 + exp(logvar_i) - logvar_i - 1 ) ]
    """
    kl_per_sample = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
    return kl_per_sample.mean()


def kl_feature_level(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Per-feature batch moment matching: for each dim d, match the batch-level
    distribution of mu and sigma to N(0,1) using squared L2 penalty.

      mean_penalty_d = ( E_n[mu[n,d]] )^2          → batch mean mu should be 0
      var_penalty_d  = ( E_n[sigma[n,d]^2] - 1 )^2 → batch mean sigma^2 should be 1

    loss = sum_d ( mean_penalty_d + var_penalty_d )

    Why this is better than kl_bottleneck for selectivity:
      - kl_bottleneck gradient w.r.t. mu[n,d] = beta * mu[n,d]
        → large mu (informative) gets large gradient → always fighting reconstruction
      - this gradient w.r.t. mu[n,d] = 2 * mu_mean_d / B
        → gradient is the SAME for all samples, scaled by the batch mean
        → selective feature (mu large for few images): mu_mean_d is small
          → gradient is small → reconstruction can dominate → feature stays active
        → always-on feature (mu nonzero for all images): mu_mean_d is large
          → gradient is large → penalised
    """
    mu_mean    = mu.mean(dim=0)             # [D]
    sigma2_mean = logvar.exp().mean(dim=0)  # [D]

    mean_penalty = mu_mean.pow(2)           # [D]
    var_penalty  = (sigma2_mean - 1).pow(2) # [D]

    return (mean_penalty + var_penalty).sum()


def kl_feature_level_log(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Decomposed sparsity-inducing KL: log penalty on mu, direct KL on sigma.

      mu_term    = sum_d log(1 + mean_n[mu[n,d]²])
                   → sparsity: active dims saturate, gradient ∝ 1/(1+mean_mu²)
      sigma_term = sum_d mean_n[sigma[n,d]² - log(sigma[n,d]²) - 1]
                   → sigma stays in (0,∞): -log term → ∞ as sigma→0 (prevents collapse)
                                           sigma² term → ∞ as sigma→∞ (prevents >1)

    Keeping sigma OUTSIDE the log prevents saturation that caused sigma→0 in the
    previous (combined) version.
    """
    mu2_mean      = mu.pow(2).mean(dim=0)                   # [D], ≥0
    kl_sigma_mean = (logvar.exp() - logvar - 1).mean(dim=0) # [D], ≥0
    return torch.log1p(mu2_mean).sum() + kl_sigma_mean.sum()


def l1_mu_loss(mu: torch.Tensor) -> torch.Tensor:
    """
    L1 penalty on encoder mean: mean_N mean_D |mu_{n,d}|.

    Adds a Laplace prior on mu alongside the Gaussian prior from KL.
    Unlike KL's gradient ∝ mu (small mu → small gradient), the L1 gradient is
    ±alpha (constant magnitude), so near-zero dims get pushed harder toward
    exactly 0, sharpening sparsity.
    """
    return mu.abs().mean()


def kl_feature_level_sq(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Per-dim squared mean-KL loss.

    For each dim d:
      f_d = mean_n[mu[n,d]²] + mean_n[sigma[n,d]² - log(sigma[n,d]²) - 1]
          = 2 * mean_n[ KL(q(z_d|x_n) || N(0,1)) ]

    loss = sum_d( f_d² )

    Gradient w.r.t. mu[n,d] ∝ mean_n[KL_d] * mu[n,d]:
      - inactive dim (mean KL ≈ 0): gradient ≈ 0 → reconstruction dominates → feature survives
      - always-on dim (mean KL large): gradient large → pushed to prior
    No sign-cancellation: mu² ≥ 0 and sigma²-log(sigma²)-1 ≥ 0 for all inputs.
    """
    mu2_mean      = mu.pow(2).mean(dim=0)                          # [D]
    kl_sigma_mean = (logvar.exp() - logvar - 1).mean(dim=0)        # [D]
    per_dim       = mu2_mean + kl_sigma_mean                       # [D], = 2*mean_n[KL_{n,d}]
    return per_dim.pow(2).sum()


class MomentSIGRegLoss(nn.Module):
    """Sliced moment-matching SIGReg kept for ablations."""

    def __init__(
        self,
        num_projections: int = 256,
        lambda_skew: float = 0.1,
        lambda_kurt: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_projections = num_projections
        self.lambda_skew = lambda_skew
        self.lambda_kurt = lambda_kurt
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, D = z.shape

        # Historical implementation centered z before sliced moments.
        z_centered = z - z.mean(dim=0, keepdim=True)

        A = torch.randn(self.num_projections, D, device=z.device, dtype=z.dtype)
        A = F.normalize(A, dim=1)
        y = z_centered @ A.T

        mean_y = y.mean(dim=0)
        var_y = y.var(dim=0, unbiased=False)

        y_c = y - mean_y.unsqueeze(0)
        std_y = (var_y + self.eps).sqrt()

        skew_y = y_c.pow(3).mean(dim=0) / (std_y.pow(3) + self.eps)
        kurt_y = y_c.pow(4).mean(dim=0) / (std_y.pow(4) + self.eps)

        mean_penalty = mean_y.pow(2).mean()
        var_penalty = (var_y - 1).pow(2).mean()
        skew_penalty = skew_y.pow(2).mean()
        kurt_penalty = (kurt_y - 3).pow(2).mean()

        return (
            mean_penalty
            + var_penalty
            + self.lambda_skew * skew_penalty
            + self.lambda_kurt * kurt_penalty
        )


class EppsPulleySIGReg(nn.Module):
    """
    Sliced Epps-Pulley SIGReg.

    Random unit directions reduce z: [B, D] to y: [B, K]. Each projected
    empirical characteristic function is matched to N(0, 1), without
    standardizing z or y.
    """

    def __init__(
        self,
        num_slices: int = 256,
        num_points: int = 33,
        t_min: float = -5.0,
        t_max: float = 5.0,
        weighted: bool = True,
        slice_chunk_size: int = 256,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.num_points = num_points
        self.t_min = t_min
        self.t_max = t_max
        self.weighted = weighted
        self.slice_chunk_size = slice_chunk_size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"Expected z with shape [B, D], got {tuple(z.shape)}")
        _, D = z.shape

        compute_dtype = torch.float32
        z_f = z.to(dtype=compute_dtype)

        # A has shape [D, K]; only directions are normalized.
        A = torch.randn(D, self.num_slices, device=z.device, dtype=compute_dtype)
        A = F.normalize(A, dim=0)
        y = z_f @ A

        t = torch.linspace(
            self.t_min,
            self.t_max,
            self.num_points,
            device=z.device,
            dtype=compute_dtype,
        )
        target_real = torch.exp(-0.5 * t.pow(2))
        weight = target_real if self.weighted else torch.ones_like(t)

        loss_sum = y.new_zeros(())
        counted = 0
        chunk_size = max(1, self.slice_chunk_size)
        for start in range(0, self.num_slices, chunk_size):
            y_chunk = y[:, start : start + chunk_size]
            yt = y_chunk[:, :, None] * t[None, None, :]

            ecf_real = torch.cos(yt).mean(dim=0)
            ecf_imag = torch.sin(yt).mean(dim=0)

            diff_real = ecf_real - target_real[None, :]
            squared_error = diff_real.pow(2) + ecf_imag.pow(2)
            chunk_loss = (weight[None, :] * squared_error).mean()

            n_chunk = y_chunk.shape[1]
            loss_sum = loss_sum + chunk_loss * n_chunk
            counted += n_chunk

        loss = loss_sum / max(counted, 1)
        return torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)


class NullSIGRegLoss(nn.Module):
    """No-op SIGReg for ablations."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z.sum() * 0.0


def build_sigreg_loss(
    sigreg_type: str = "epps_pulley",
    num_projections: int = 256,
    ep_num_points: int = 33,
    ep_t_min: float = -5.0,
    ep_t_max: float = 5.0,
    ep_weighted: bool = True,
    ep_slice_chunk_size: int = 256,
) -> nn.Module:
    """Factory used by training scripts to select the SIGReg variant."""
    if sigreg_type == "moment":
        return MomentSIGRegLoss(num_projections=num_projections)
    if sigreg_type == "epps_pulley":
        return EppsPulleySIGReg(
            num_slices=num_projections,
            num_points=ep_num_points,
            t_min=ep_t_min,
            t_max=ep_t_max,
            weighted=ep_weighted,
            slice_chunk_size=ep_slice_chunk_size,
        )
    if sigreg_type == "none":
        return NullSIGRegLoss()
    raise ValueError(f"Unknown sigreg_type: {sigreg_type}")


def sigreg_loss(
    z: torch.Tensor,
    num_projections: int = 256,
    bandwidths: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    sigreg_type: str = "epps_pulley",
    ep_num_points: int = 33,
) -> torch.Tensor:
    """
    Backward-compatible functional SIGReg entry point.

    The default is now sliced Epps-Pulley characteristic-function matching.
    Pass sigreg_type="moment" to reproduce the older moment ablation.
    """
    del bandwidths, eps
    return build_sigreg_loss(
        sigreg_type=sigreg_type,
        num_projections=num_projections,
        ep_num_points=ep_num_points,
    )(z)
