from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2WithRegistersModel


def _load_ckpt(path: str | Path) -> dict:
    load_sig = inspect.signature(torch.load)
    kwargs: dict = {"map_location": "cpu"}
    if "weights_only" in load_sig.parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


class VaeSigregConditioner(nn.Module):
    """DINO CLS token -> frozen VAE-SIGREG encoder -> condition vector.

    Architecture params are auto-inferred from the checkpoint's saved args.
    Override via constructor if the checkpoint predates args saving.
    """

    def __init__(
        self,
        vae_ckpt_path: str,
        vae_src_path: str,
        encoder_config_path: str,
        dinov2_path: str,
        encoder_input_size: int = 224,
        input_dim: int = 768,
        latent_dim: int = 3072,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        sample_temperature: float = 1.0,
    ):
        super().__init__()
        self.sample_temperature = float(sample_temperature)

        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.register_buffer(
            "mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(proc.image_std).view(1, 3, 1, 1), persistent=False
        )
        self.encoder_input_size = int(encoder_input_size)

        self.dino = Dinov2WithRegistersModel.from_pretrained(dinov2_path)
        self.dino.requires_grad_(False)

        vae_src = Path(vae_src_path)
        if str(vae_src) not in sys.path:
            sys.path.insert(0, str(vae_src))
        from vae_sigreg.model import OvercompleteVariationalAE  # type: ignore

        ckpt = _load_ckpt(vae_ckpt_path)
        saved_args = ckpt.get("args", {})
        input_dim  = int(saved_args.get("input_dim",  input_dim))
        latent_dim = int(saved_args.get("latent_dim", latent_dim))
        hidden_dim_val = saved_args.get("hidden_dim", hidden_dim)
        hidden_dim = int(hidden_dim_val) if hidden_dim_val is not None else None
        num_layers = int(saved_args.get("num_layers", num_layers))

        self.vae = OvercompleteVariationalAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.vae.load_state_dict(ckpt.get("model", ckpt))
        self.vae.requires_grad_(False)

        self._cond_dim = latent_dim

    @property
    def cond_dim(self) -> int:
        return self._cond_dim

    @torch.no_grad()
    def images_to_cls(self, images: torch.Tensor) -> torch.Tensor:
        _, _, h, w = images.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            images = nn.functional.interpolate(
                images,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )
        x = (images - self.mean.to(images.device)) / self.std.to(images.device)
        return self.dino(x).last_hidden_state[:, 0, :]

    @torch.no_grad()
    def condition_from_cls(self, cls: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.vae.encode(cls)
        std = (0.5 * logvar).exp()
        return mu + self.sample_temperature * torch.randn_like(std) * std

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.condition_from_cls(self.images_to_cls(images))

    def sample_prior(self, batch_size: int, device: torch.device | str, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Sample unconditional condition vectors from the VAE prior."""
        dtype = dtype or next(self.vae.parameters()).dtype
        return torch.randn(batch_size, self.cond_dim, device=device, dtype=dtype) * self.sample_temperature
