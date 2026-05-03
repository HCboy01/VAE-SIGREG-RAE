#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from torch import amp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2WithRegistersModel


def _add_sys_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class CenterCropTransform:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, pil_image):
        s = self.image_size
        while min(*pil_image.size) >= 2 * s:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
        scale = s / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
        arr = np.array(pil_image)
        cy = (arr.shape[0] - s) // 2
        cx = (arr.shape[1] - s) // 2
        return Image.fromarray(arr[cy: cy + s, cx: cx + s])


class UnlabeledImageDataset(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, image_size: int, hflip: bool):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.root}")
        self.files = sorted(
            p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in self.IMG_EXTS
        )
        if len(self.files) == 0:
            raise RuntimeError(f"No images found under: {self.root}")
        self.hflip = bool(hflip)
        self.transform = transforms.Compose([CenterCropTransform(image_size), transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.hflip:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return self.transform(img), str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute DINO CLS tokens for VAE-SIGREG-RAE training.")
    parser.add_argument("--config", type=str, required=True, help="VAE-SIGREG-RAE training YAML config")
    parser.add_argument("--data-path", type=str, required=True, help="Image directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write cls.npy, metadata.json, paths.jsonl")
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--storage-dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--num-augs", type=int, default=2, help="Offline copies per image; odd aug ids are horizontal flips.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing cache directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_augs < 1:
        raise ValueError("--num-augs must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script currently expects CUDA.")

    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent
    _add_sys_path(project_root / "vendor" / "rae_src")

    from utils.train_utils import parse_configs

    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output dir is not empty: {out_dir}. Pass --overwrite to replace cache files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    full_cfg = OmegaConf.load(args.config)
    rae_config, *_ = parse_configs(full_cfg)
    if rae_config is None:
        raise ValueError("Config must contain stage_1.")
    vae_cond_cfg = full_cfg.get("vae_condition", {})
    s1_params = rae_config.get("params", {})
    encoder_params = s1_params.get("encoder_params", {})
    encoder_config_path = str(vae_cond_cfg.get("encoder_config_path", s1_params.get("encoder_config_path")))
    dinov2_path = str(vae_cond_cfg.get("dinov2_path", encoder_params.get("dinov2_path", encoder_config_path)))
    encoder_input_size = int(vae_cond_cfg.get("encoder_input_size", s1_params.get("encoder_input_size", 224)))

    proc = AutoImageProcessor.from_pretrained(encoder_config_path)
    mean = torch.tensor(proc.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device=device).view(1, 3, 1, 1)
    dino = Dinov2WithRegistersModel.from_pretrained(dinov2_path).to(device)
    dino.eval().requires_grad_(False)

    probe = UnlabeledImageDataset(args.data_path, args.image_size, hflip=False)
    total = len(probe) * args.num_augs
    use_amp = args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    np_dtype = np.float16 if args.storage_dtype == "fp16" else np.float32

    probe_loader = DataLoader(probe, batch_size=min(args.batch_size, len(probe)), shuffle=False, num_workers=args.workers)
    first_images, _ = next(iter(probe_loader))
    first_images = first_images.to(device, non_blocking=True)
    with torch.no_grad(), amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        if first_images.shape[-1] != encoder_input_size or first_images.shape[-2] != encoder_input_size:
            first_images = nn.functional.interpolate(
                first_images, size=(encoder_input_size, encoder_input_size), mode="bicubic", align_corners=False
            )
        first_cls = dino((first_images - mean) / std).last_hidden_state[:, 0, :]

    cls_shape = tuple(first_cls.shape[1:])
    cls_mm = np.lib.format.open_memmap(out_dir / "cls.npy", mode="w+", dtype=np_dtype, shape=(total, *cls_shape))
    metadata = {
        "format": "dino_cls_cache_v1",
        "source_data_path": str(Path(args.data_path).resolve()),
        "config": str(Path(args.config).resolve()),
        "image_size": args.image_size,
        "encoder_input_size": encoder_input_size,
        "encoder_config_path": encoder_config_path,
        "dinov2_path": dinov2_path,
        "num_source_images": len(probe),
        "num_augs": args.num_augs,
        "num_samples": total,
        "cls_shape": [total, *cls_shape],
        "storage_dtype": args.storage_dtype,
        "precision": args.precision,
        "augmentation": "center_crop + deterministic horizontal flip on odd aug ids",
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    paths_f = (out_dir / "paths.jsonl").open("w")
    write_idx = 0
    t0 = time()
    try:
        for aug_idx in range(args.num_augs):
            hflip = (aug_idx % 2) == 1
            dataset = UnlabeledImageDataset(args.data_path, args.image_size, hflip=hflip)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
            )
            for step, (images, paths) in enumerate(loader, start=1):
                images = images.to(device, non_blocking=True)
                with torch.no_grad(), amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    if images.shape[-1] != encoder_input_size or images.shape[-2] != encoder_input_size:
                        images = nn.functional.interpolate(
                            images, size=(encoder_input_size, encoder_input_size), mode="bicubic", align_corners=False
                        )
                    cls = dino((images - mean) / std).last_hidden_state[:, 0, :]
                bsz = cls.shape[0]
                cls_mm[write_idx: write_idx + bsz] = cls.detach().float().cpu().numpy().astype(np_dtype, copy=False)
                for path in paths:
                    paths_f.write(json.dumps({"path": path, "aug_idx": aug_idx, "hflip": hflip}) + "\n")
                write_idx += bsz
                if step % 100 == 0:
                    it_s = write_idx / max(time() - t0, 1e-6)
                    eta_min = (total - write_idx) / max(it_s, 1e-6) / 60.0
                    print(
                        f"[cache-dino-cls] aug={aug_idx + 1}/{args.num_augs} step={step:05d} "
                        f"written={write_idx}/{total} ({100.0 * write_idx / total:.1f}%) eta={eta_min:.1f}m",
                        flush=True,
                    )
    finally:
        paths_f.close()
        cls_mm.flush()

    print(f"[done] wrote DINO CLS cache to {out_dir} samples={write_idx} cls_shape={cls_mm.shape}", flush=True)


if __name__ == "__main__":
    main()
