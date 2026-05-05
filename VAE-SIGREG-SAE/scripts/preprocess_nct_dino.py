#!/usr/bin/env python3
"""Extract DINOv2 CLS features from NCT-CRC-HE-100K (folder of images) and save as .bin."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2WithRegistersModel


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths: list[Path], encoder_input_size: int):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((encoder_input_size, encoder_input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            img = img.convert("RGB")
        return self.transform(img), str(self.image_paths[idx])


def write_split(
    *,
    name: str,
    image_paths: list[Path],
    output_dir: Path,
    model: Dinov2WithRegistersModel,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    encoder_input_size: int,
    batch_size: int,
    num_workers: int,
    precision: str,
):
    feature_path = output_dir / f"{name}_features.bin"
    shape_path = output_dir / f"{name}_features_shape.npy"
    paths_path = output_dir / f"{name}_paths.txt"

    use_amp = precision in {"bf16", "fp16"} and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    dataset = ImageFolderDataset(image_paths, encoder_input_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
    )

    n = len(image_paths)
    written = 0
    feat_dim = None
    t0 = time()
    with feature_path.open("wb") as features_f, paths_path.open("w") as paths_f:
        for step, (images, paths) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                cls = model((images - mean) / std).last_hidden_state[:, 0, :]
            features = cls.float().cpu().numpy().astype(np.float32, copy=False)
            features.tofile(features_f)
            for path in paths:
                paths_f.write(f"{path}\n")
            written += features.shape[0]
            if feat_dim is None:
                feat_dim = features.shape[1]
                np.save(shape_path, np.array([n, feat_dim], dtype=np.int64))
            if step % 50 == 0 or written >= n:
                rate = written / max(time() - t0, 1e-6)
                eta_min = (n - written) / max(rate, 1e-6) / 60.0
                print(f"[{name}] {written}/{n} ({100.0*written/n:.1f}%) rate={rate:.0f}/s eta={eta_min:.1f}m", flush=True)

    np.save(shape_path, np.array([written, feat_dim], dtype=np.int64))
    print(f"[{name}] done: {written} samples, dim={feat_dim}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True, help="Root folder (contains class subfolders)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="facebook/dinov2-with-registers-base")
    parser.add_argument("--encoder-input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected = [output_dir / "train_features.bin", output_dir / "eval_features.bin"]
    if not args.overwrite and any(p.exists() for p in expected):
        raise FileExistsError(f"Output files already exist in {output_dir}. Pass --overwrite to replace.")

    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    all_paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in exts)
    if not all_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    print(f"Found {len(all_paths)} images", flush=True)

    rng = random.Random(args.seed)
    shuffled = all_paths[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_fraction))
    val_paths = shuffled[:n_val]
    train_paths = shuffled[n_val:]
    print(f"Split: train={len(train_paths)} eval={len(val_paths)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(args.model)
    mean = torch.tensor(proc.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device=device).view(1, 3, 1, 1)
    model = Dinov2WithRegistersModel.from_pretrained(args.model).to(device)
    model.eval().requires_grad_(False)

    meta = {
        "image_dir": str(image_dir.resolve()),
        "model": args.model,
        "encoder_input_size": args.encoder_input_size,
        "precision": args.precision,
        "total": len(all_paths),
        "train": len(train_paths),
        "eval": len(val_paths),
        "seed": args.seed,
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    common = dict(
        output_dir=output_dir, model=model, mean=mean, std=std, device=device,
        encoder_input_size=args.encoder_input_size, batch_size=args.batch_size,
        num_workers=args.num_workers, precision=args.precision,
    )
    write_split(name="train", image_paths=train_paths, **common)
    write_split(name="eval", image_paths=val_paths, **common)
    print(f"Done. Features saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
