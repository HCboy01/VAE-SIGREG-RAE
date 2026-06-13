#!/usr/bin/env python3
"""
한 개 feature dimension의 mean/std 측정.

사용법:
    python scripts/probe_feature_dim.py \
        --ckpt checkpoints/s1_grid_b1e-3_l100_lindec/best.pt \
        --img_dir /path/to/images \
        --feature_dim 42 \
        --n_samples 100
"""
import argparse
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
    p.add_argument("--feature_dim", type=int, required=True)
    p.add_argument("--n_samples",   type=int, default=100)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--dino_path",   type=str, default="facebook/dinov2-with-registers-base")
    args = p.parse_args()

    # 이미지 수집
    paths = sorted(p for p in Path(args.img_dir).rglob("*")
                   if p.is_file() and p.suffix.lower() in IMG_EXTS)[:args.n_samples]
    print(f"이미지 {len(paths)}개  |  feature dim: {args.feature_dim}", flush=True)

    # 모델 로드
    print("DINO 로드...", flush=True)
    dino, dino_mean, dino_std = load_dino(args.dino_path)
    print("VAE 로드...", flush=True)
    vae = load_vae(args.ckpt)

    to_tensor = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # 인코딩
    dim_values = []
    for i in range(0, len(paths), args.batch_size):
        batch = [to_tensor(Image.open(p).convert("RGB")) for p in paths[i:i+args.batch_size]]
        imgs  = torch.stack(batch).to(DEVICE)

        with torch.no_grad():
            if imgs.shape[-1] != 224:
                imgs = F.interpolate(imgs, (224, 224), mode="bicubic", align_corners=False)
            x = (imgs - dino_mean) / dino_std
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEVICE.type == "cuda")):
                cls = dino(x).last_hidden_state[:, 0, :].float()
                mu, _ = vae.encode(cls)

        dim_values.append(mu[:, args.feature_dim].float().cpu())
        print(f"  {min(i+args.batch_size, len(paths))}/{len(paths)}", flush=True)

    vals = torch.cat(dim_values)
    print(f"\n결과 (mu, dim={args.feature_dim}, n={len(vals)})")
    print(f"  mean = {vals.mean():.4f}")
    print(f"  std  = {vals.std():.4f}")
    print(f"  min  = {vals.min():.4f}")
    print(f"  max  = {vals.max():.4f}")


if __name__ == "__main__":
    main()
