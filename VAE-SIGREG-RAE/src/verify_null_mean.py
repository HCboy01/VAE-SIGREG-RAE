#!/usr/bin/env python3
"""Verify if null_cond ≈ E[z_cond from real images].

Computes μ_real = mean of z_cond over real images,
then measures cosine similarity and L2 distance with null_cond.

Usage:
    cd /scratch/x3411a10/VAE-SIGREG/VAE-SIGREG-RAE
    CUDA_VISIBLE_DEVICES=X python src/verify_null_mean.py \
        --ckpt ckpts/.../best.pt \
        --real-path /scratch/x3411a10/datasets/ffhq256/imagefolder/val
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _add_sys_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_ckpt(path):
    sig = inspect.signature(torch.load)
    kw = {"map_location": "cpu"}
    if "mmap" in sig.parameters: kw["mmap"] = True
    if "weights_only" in sig.parameters: kw["weights_only"] = False
    return torch.load(path, **kw)


class CenterCropTransform:
    def __init__(self, s=256):
        self.s = s
    def __call__(self, img):
        import numpy as np
        s = self.s
        while min(*img.size) >= 2 * s:
            img = img.resize(tuple(x // 2 for x in img.size), resample=Image.BOX)
        scale = s / min(*img.size)
        img = img.resize(tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC)
        arr = np.array(img)
        cy, cx = (arr.shape[0] - s) // 2, (arr.shape[1] - s) // 2
        return Image.fromarray(arr[cy:cy+s, cx:cx+s])


class ImageDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    def __init__(self, root, transform=None):
        self.files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in self.EXTS)
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        with Image.open(self.files[idx]) as img:
            img = img.convert("RGB")
        if self.transform: img = self.transform(img)
        return img


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True)
    p.add_argument("--real-path",  required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--max-samples", type=int, default=5000)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent
    _add_sys_path(project_root / "vendor" / "rae_src")
    _add_sys_path(project_root / "src")

    from vae_rae.conditioning import VaeSigregConditioner

    ckpt = _load_ckpt(args.ckpt)
    cfg  = ckpt["config"]
    vc   = cfg["vae_condition"]
    s1   = cfg["stage_1"].get("params", {})

    print("[info] loading conditioner ...", flush=True)
    conditioner = VaeSigregConditioner(
        encoder_config_path=str(vc.get("encoder_config_path", s1.get("encoder_config_path"))),
        dinov2_path=str(vc.get("dinov2_path", s1.get("encoder_params", {}).get("dinov2_path"))),
        encoder_input_size=int(vc.get("encoder_input_size", 224)),
        vae_ckpt_path=str(vc["vae_ckpt"]),
        vae_src_path=str(vc["vae_src_path"]),
        sample_temperature=float(vc.get("sample_temperature", 1.0)),
    ).to(device)
    conditioner.eval().requires_grad_(False)
    cond_dim = conditioner.cond_dim

    state = ckpt.get("ema", ckpt.get("model"))
    null_cond = state["null_cond"].to(device).float().squeeze(0)  # [cond_dim]
    print(f"[info] null_cond: norm={null_cond.norm():.4f}  mean={null_cond.mean():.4f}  std={null_cond.std():.4f}", flush=True)

    transform = transforms.Compose([CenterCropTransform(224), transforms.ToTensor()])
    dataset   = ImageDataset(args.real_path, transform=transform)
    n = min(len(dataset), args.max_samples)
    dataset.files = dataset.files[:n]
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, pin_memory=True)

    print(f"[info] computing z_cond for {n} real images ...", flush=True)
    sum_z   = torch.zeros(cond_dim, device=device)
    sum_z2  = torch.zeros(cond_dim, device=device)
    count   = 0

    with torch.no_grad():
        for i, imgs in enumerate(loader):
            imgs = imgs.to(device)
            z = conditioner(imgs).float()  # [B, cond_dim]
            sum_z  += z.sum(0)
            sum_z2 += (z ** 2).sum(0)
            count  += z.shape[0]
            if (i + 1) % 10 == 0:
                print(f"  {count}/{n}", flush=True)

    mu_real = sum_z / count
    var_real = sum_z2 / count - mu_real ** 2
    std_real = var_real.clamp(min=0).sqrt()

    print(f"\n[μ_real] norm={mu_real.norm():.4f}  mean={mu_real.mean():.4f}  std={mu_real.std():.4f}", flush=True)
    print(f"[μ_real] per-dim std (avg): {std_real.mean():.4f}", flush=True)

    cos_sim = F.cosine_similarity(null_cond.unsqueeze(0), mu_real.unsqueeze(0)).item()
    l2_dist = (null_cond - mu_real).norm().item()
    l2_null = null_cond.norm().item()
    l2_mu   = mu_real.norm().item()

    print(f"\n=== Results ===")
    print(f"cosine_similarity(null_cond, μ_real) = {cos_sim:.6f}")
    print(f"||null_cond - μ_real||               = {l2_dist:.4f}")
    print(f"||null_cond||                         = {l2_null:.4f}")
    print(f"||μ_real||                            = {l2_mu:.4f}")
    print(f"relative distance = {l2_dist / l2_null:.4f}  (0=identical, 1=orthogonal scale)")

    if cos_sim > 0.9:
        print("\n[결론] null_cond ≈ μ_real 가설 강하게 지지됨")
    elif cos_sim > 0.5:
        print("\n[결론] 방향은 비슷하나 완전히 일치하지는 않음")
    else:
        print("\n[결론] null_cond가 μ_real 방향을 근사하지 않음 → 가설 기각")


if __name__ == "__main__":
    main()
