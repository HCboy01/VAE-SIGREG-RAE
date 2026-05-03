from typing import Optional

import torch
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


def sigreg_loss(
    z: torch.Tensor,
    num_projections: int = 256,
    bandwidths: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    SIGReg: regularizes the aggregate posterior q(z) = ∫q(z|x)p(x)dx
    toward an isotropic Gaussian N(0, I) via sliced moment matching.

    Random projection directions reduce this to a set of 1D problems:
    each projected distribution y_j should look like N(0, 1).

    Gradients flow back to encoder — do NOT detach z.

    Penalties:
      mean²          -> 0        (zero mean)
      (var - 1)²     -> 0        (unit variance)
      skew²          -> 0        (symmetric)
      (kurt - 3)²    -> 0        (mesokurtic, weight 0.1)
    """
    B, D = z.shape

    # Center across batch so projections measure shape, not offset
    z_centered = z - z.mean(dim=0, keepdim=True)

    # Random unit-norm directions on the same device/dtype as z
    A = torch.randn(num_projections, D, device=z.device, dtype=z.dtype)
    A = F.normalize(A, dim=1)

    # Project: [B, num_projections]
    y = z_centered @ A.T

    mean_y = y.mean(dim=0)                        # [P]
    var_y = y.var(dim=0, unbiased=False)          # [P]

    # Higher moments computed from centered y
    y_c = y - mean_y.unsqueeze(0)
    std_y = (var_y + eps).sqrt()

    skew_y = y_c.pow(3).mean(dim=0) / (std_y.pow(3) + eps)
    kurt_y = y_c.pow(4).mean(dim=0) / (std_y.pow(4) + eps)  # raw kurtosis

    mean_penalty = mean_y.pow(2).mean()
    var_penalty = (var_y - 1).pow(2).mean()
    skew_penalty = skew_y.pow(2).mean()
    kurt_penalty = (kurt_y - 3).pow(2).mean()

    return mean_penalty + var_penalty + 0.1 * skew_penalty + 0.1 * kurt_penalty
