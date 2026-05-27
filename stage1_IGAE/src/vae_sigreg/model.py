from typing import Dict, Optional

import torch
import torch.nn as nn


def _make_mlp_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
    )


class OvercompleteVariationalAE(nn.Module):
    """
    Overcomplete VAE: latent_dim may be larger than input_dim.

    Encoder maps x -> (mu, logvar) via MLP with LayerNorm + GELU.
    Decoder maps z -> x_hat. When linear_decoder=True (default), decoder is a
    single affine layer z -> x_hat, which prevents dead features by ensuring
    every latent dimension receives reconstruction gradients directly.
    When linear_decoder=False, decoder uses the same MLP structure as before.

    hidden_dim defaults to input_dim * 4 (e.g. 768 -> 3072).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        linear_decoder: bool = True,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 4

        self.latent_dim = latent_dim

        # --- Encoder backbone: input_dim -> hidden_dim (x num_layers) ---
        enc_layers: list[nn.Module] = []
        enc_in = input_dim
        for _ in range(num_layers):
            enc_layers.append(_make_mlp_block(enc_in, hidden_dim))
            enc_in = hidden_dim
        self.encoder_backbone = nn.Sequential(*enc_layers)

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # --- Decoder ---
        if linear_decoder:
            self.decoder = nn.Linear(latent_dim, input_dim)
        else:
            dec_layers: list[nn.Module] = []
            dec_in = latent_dim
            for _ in range(num_layers):
                dec_layers.append(_make_mlp_block(dec_in, hidden_dim))
                dec_in = hidden_dim
            dec_layers.append(nn.Linear(hidden_dim, input_dim))
            self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor):
        h = self.encoder_backbone(x)
        mu = self.mu_head(h)
        # clamp logvar for numerical stability
        logvar = self.logvar_head(h).clamp(-10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "mu": mu, "logvar": logvar, "z": z}
