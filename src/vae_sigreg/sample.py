import torch

from .model import OvercompleteVariationalAE


@torch.no_grad()
def sample_from_prior(
    model: OvercompleteVariationalAE,
    num_samples: int,
    latent_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample z_prior ~ N(0, I) and decode to embedding space.
    Returns x_hat of shape [num_samples, input_dim].
    """
    model.eval()
    z_prior = torch.randn(num_samples, latent_dim, device=device)
    return model.decode(z_prior)
