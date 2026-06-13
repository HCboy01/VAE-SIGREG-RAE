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


def sigma_feature_deviation_loss(
    logvar: torch.Tensor,
    target: float = 1.0,
    metric: str = "l2",
    only_below: bool = False,
) -> torch.Tensor:
    """
    Batch feature-axis sigma regularizer.

    For each latent feature d, compute mean_B[sigma_{b,d}] over the current
    physical batch and penalize its deviation from target, usually 1.0.
    This keeps posterior stds close to the prior scale at the feature level.
    """
    if logvar.ndim != 2:
        raise ValueError(f"Expected logvar with shape [B, D], got {tuple(logvar.shape)}")
    sigma_mean = (0.5 * logvar.float()).exp().mean(dim=0)
    dev = sigma_mean - float(target)
    if only_below:
        dev = torch.clamp(dev, max=0.0)
    if metric == "l1":
        return dev.abs().mean()
    if metric == "l2":
        return dev.pow(2).mean()
    raise ValueError(f"Unknown sigma deviation metric: {metric}")


def sigma_low_fraction_loss(
    logvar: torch.Tensor,
    threshold: float = 0.8,
    target_fraction: float = 0.1,
    min_fraction: float = 0.0,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Soft feature-wise low-sigma fraction band regularizer.

    For each latent feature d, compute:
        low_score_bd = sigmoid((threshold - sigma_bd) / temperature)
        low_frac_d = mean_B[low_score_bd]

    Then penalize low_frac_d outside [min_fraction, target_fraction]. This
    discourages global confidence while also pushing dead features to become
    selectively confident on at least a small fraction of images.
    """
    if logvar.ndim != 2:
        raise ValueError(f"Expected logvar with shape [B, D], got {tuple(logvar.shape)}")
    sigma = (0.5 * logvar.float()).exp()
    temp = max(float(temperature), 1e-6)
    low_score = torch.sigmoid((float(threshold) - sigma) / temp)
    low_frac = low_score.mean(dim=0)
    over = torch.relu(low_frac - float(target_fraction))
    under = torch.relu(float(min_fraction) - low_frac)
    return (over + under).mean()


def sigma_low_tail_target_loss(
    logvar: torch.Tensor,
    tail_fraction: float = 0.05,
    target: float = 0.5,
    metric: str = "l2",
) -> torch.Tensor:
    """
    Pull each feature's low-sigma tail toward a target value.

    For every latent feature d, take the lowest tail_fraction of sigma over the
    current batch, average those selected values, and penalize deviation from
    target. This directly encourages each feature to have a small subset of
    images with genuinely low confidence scale, e.g. sigma around 0.5.
    """
    if logvar.ndim != 2:
        raise ValueError(f"Expected logvar with shape [B, D], got {tuple(logvar.shape)}")
    sigma = (0.5 * logvar.float()).exp()
    B = sigma.shape[0]
    k = max(1, min(B, int(round(B * float(tail_fraction)))))
    tail = torch.topk(sigma, k=k, dim=0, largest=False).values
    tail_mean = tail.mean(dim=0)
    dev = tail_mean - float(target)
    if metric == "l1":
        return dev.abs().mean()
    if metric == "l2":
        return dev.pow(2).mean()
    raise ValueError(f"Unknown sigma low-tail metric: {metric}")


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


def build_sigreg_loss(
    num_projections: int = 256,
    ep_num_points: int = 33,
    ep_t_min: float = -5.0,
    ep_t_max: float = 5.0,
    ep_weighted: bool = True,
    ep_slice_chunk_size: int = 256,
) -> EppsPulleySIGReg:
    """Build the default D-space Epps-Pulley SIGReg loss."""
    return EppsPulleySIGReg(
        num_slices=num_projections,
        num_points=ep_num_points,
        t_min=ep_t_min,
        t_max=ep_t_max,
        weighted=ep_weighted,
        slice_chunk_size=ep_slice_chunk_size,
    )


def sigreg_loss(
    z: torch.Tensor,
    num_projections: int = 256,
    ep_num_points: int = 33,
) -> torch.Tensor:
    return build_sigreg_loss(
        num_projections=num_projections,
        ep_num_points=ep_num_points,
    )(z)
