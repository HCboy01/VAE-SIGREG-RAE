#!/usr/bin/env python3
"""
단일 feature dimension의 100개 이미지별 μ값 출력.

Usage:
    python scripts/probe_single_feature.py \
        --ckpt checkpoints/s1_grid_b1e-3_l100_lindec/best.pt \
        --img_dir /scratch/x3411a10/datasets/ffhq256/imagefolder/train/images \
        --feature_dim 1 \
        --n_samples 100
"""
import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vae_sigreg.model import OvercompleteVariationalAE

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dino(dino_path="facebook/dinov2-with-registers-base"):
    from transformers import AutoImageProcessor, Dinov2WithRegistersModel
    proc  = AutoImageProcessor.from_pretrained(dino_path)
    model = Dinov2WithRegistersModel.from_pretrained(dino_path).to(DEVICE)
    model.eval().requires_grad_(False)
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to(DEVICE)
    std  = torch.tensor(proc.image_std).view(1, 3, 1, 1).to(DEVICE)
    return model, mean, std


def load_vae(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    model = OvercompleteVariationalAE(
        input_dim  = int(a.get("input_dim",  768)),
        latent_dim = int(a.get("latent_dim", 6144)),
        hidden_dim = a.get("hidden_dim", None),
        num_layers = int(a.get("num_layers", 4)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval().requires_grad_(False)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--img_dir",     required=True)
    p.add_argument("--feature_dim", type=int, default=1)
    p.add_argument("--n_samples",   type=int, default=100)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--dino_path",   default="facebook/dinov2-with-registers-base")
    args = p.parse_args()

    random.seed(args.seed)
    paths = sorted(p for p in Path(args.img_dir).rglob("*")
                   if p.is_file() and p.suffix.lower() in IMG_EXTS)
    paths = random.sample(paths, min(args.n_samples, len(paths)))
    print(f"이미지 {len(paths)}개  |  feature dim: {args.feature_dim}\n")

    dino, dino_mean, dino_std = load_dino(args.dino_path)
    vae = load_vae(args.ckpt)

    to_tensor = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    all_vals = []
    for i, path in enumerate(paths):
        img = to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            x = (img - dino_mean) / dino_std
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEVICE.type == "cuda")):
                cls = dino(x).last_hidden_state[:, 0, :].float()
                mu, logvar = vae.encode(cls)
                std = (0.5 * logvar).exp()
                z = mu + std * torch.randn_like(std)

        mu_val  = mu[0, args.feature_dim].item()
        std_val = std[0, args.feature_dim].item()
        all_vals.append(mu_val)
        print(f"[{i+1:3d}/100]  평균(μ): {mu_val:+.4f}   표준편차(σ): {std_val:.4f}")

    import numpy as np
    vals = np.array(all_vals)
    print(f"\n{'='*45}")
    print(f"전체 100개 요약  (feature dim={args.feature_dim})")
    print(f"  μ의 평균:   {vals.mean():+.4f}")
    print(f"  μ의 std:    {vals.std():.4f}")
    print(f"  μ의 min:    {vals.min():+.4f}")
    print(f"  μ의 max:    {vals.max():+.4f}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
