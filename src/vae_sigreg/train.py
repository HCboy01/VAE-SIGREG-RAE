from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from .losses import kl_bottleneck_loss, reconstruction_loss, sigreg_loss
from .model import OvercompleteVariationalAE


def _extract_x(batch) -> torch.Tensor:
    """Support dataloaders that return a plain tensor or a tuple/list."""
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def _compute_all_losses(
    outputs: Dict[str, torch.Tensor],
    x: torch.Tensor,
    beta_kl: float,
    lambda_sigreg: float,
    num_projections: int,
) -> Dict[str, torch.Tensor]:
    x_hat = outputs["x_hat"]
    mu = outputs["mu"]
    logvar = outputs["logvar"]
    z = outputs["z"]

    rec = reconstruction_loss(x_hat, x)
    kl = kl_bottleneck_loss(mu, logvar)
    sig = sigreg_loss(z, num_projections=num_projections)
    total = rec + beta_kl * kl + lambda_sigreg * sig

    # Diagnostics (no grad needed)
    with torch.no_grad():
        z_mean_abs = z.mean(dim=0).abs().mean()
        z_var_err = (z.var(dim=0, unbiased=False) - 1).abs().mean()

        # Off-diagonal covariance on a small subset of dims
        n_sub = min(64, z.shape[1])
        z_sub = z[:, :n_sub]
        z_sub_c = z_sub - z_sub.mean(dim=0)
        cov = (z_sub_c.T @ z_sub_c) / max(z_sub.shape[0] - 1, 1)
        off_mask = ~torch.eye(n_sub, dtype=torch.bool, device=z.device)
        off_diag_cov = cov[off_mask].abs().mean()

    return {
        "rec_loss": rec,
        "kl_loss": kl,
        "sigreg_loss": sig,
        "total_loss": total,
        # encoder health
        "mu_sq_mean": mu.pow(2).mean().detach(),
        "exp_logvar_mean": logvar.exp().mean().detach(),
        # latent distribution quality
        "z_mean_abs": z_mean_abs,
        "z_var_err": z_var_err,
        "z_off_diag_cov": off_diag_cov,
    }


def train_one_epoch(
    model: OvercompleteVariationalAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta_kl: float = 1e-4,
    lambda_sigreg: float = 1e-2,
    num_projections: int = 256,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    accum: Dict[str, float] = {}
    n_batches = 0

    iterable = tqdm(dataloader, desc="Train", leave=False) if _HAS_TQDM else dataloader
    for batch in iterable:
        x = _extract_x(batch).to(device)

        outputs = model(x)
        metrics = _compute_all_losses(outputs, x, beta_kl, lambda_sigreg, num_projections)

        optimizer.zero_grad()
        metrics["total_loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in accum.items()}


@torch.no_grad()
def eval_one_epoch(
    model: OvercompleteVariationalAE,
    dataloader: DataLoader,
    device: torch.device,
    beta_kl: float = 1e-4,
    lambda_sigreg: float = 1e-2,
    num_projections: int = 256,
) -> Dict[str, float]:
    model.eval()
    accum: Dict[str, float] = {}
    n_batches = 0

    iterable = tqdm(dataloader, desc="Eval", leave=False) if _HAS_TQDM else dataloader
    for batch in iterable:
        x = _extract_x(batch).to(device)

        outputs = model(x)
        metrics = _compute_all_losses(outputs, x, beta_kl, lambda_sigreg, num_projections)

        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v.item()
        n_batches += 1

    return {k: v / n_batches for k, v in accum.items()}
