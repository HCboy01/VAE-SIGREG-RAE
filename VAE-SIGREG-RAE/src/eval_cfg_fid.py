#!/usr/bin/env python3
"""FID/paired-FID sweep for VAE-SIGREG-conditioned CFG sampling."""
from __future__ import annotations

import argparse
import csv
import gc
import inspect
import math
import shutil
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import amp
from torch.utils.data import DataLoader, Dataset
from torch_fidelity import calculate_metrics
from torchvision import transforms
from torchvision.utils import make_grid, save_image


def _add_sys_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_ckpt(path: str | Path) -> dict:
    sig = inspect.signature(torch.load)
    kwargs: dict = {"map_location": "cpu"}
    if "mmap" in sig.parameters:
        kwargs["mmap"] = True
    if "weights_only" in sig.parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


class CenterCropTransform:
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, pil_image: Image.Image) -> Image.Image:
        import numpy as np

        s = self.image_size
        while min(*pil_image.size) >= 2 * s:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
        scale = s / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
        arr = np.array(pil_image)
        cy = (arr.shape[0] - s) // 2
        cx = (arr.shape[1] - s) // 2
        return Image.fromarray(arr[cy : cy + s, cx : cx + s])


class UnlabeledImageDataset(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str | Path, transform=None, return_path: bool = False):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.root}")
        self.transform = transform
        self.return_path = bool(return_path)
        self.files = sorted(p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in self.IMG_EXTS)
        if len(self.files) == 0:
            raise RuntimeError(f"No images found under: {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.return_path:
            return img, str(path)
        return img, 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CFG scales for a VAE-SIGREG-conditioned RAE checkpoint.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--config", type=str, default="", help="YAML config. Defaults to config embedded in checkpoint.")
    p.add_argument("--real-path", type=str, required=True, help="Real image directory for distribution FID.")
    p.add_argument("--cond-path", type=str, default="", help="Condition image directory. Defaults to --real-path.")
    p.add_argument("--out-dir", type=str, default="", help="Default: <ckpt_dir>/cfg_fid")
    p.add_argument("--cfg-scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    p.add_argument("--cfg-t-min", type=float, default=0.0)
    p.add_argument("--cfg-t-max", type=float, default=1.0)
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sampler-method", type=str, default="euler")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--keep-samples", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-pair-grid", action="store_true")
    p.add_argument("--max-grid-pairs", type=int, default=32)
    p.add_argument("--add-null", action="store_true", help="Add null_cond to image-derived z_cond before conditioning")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if any(scale > 1.0 for scale in args.cfg_scales) and not (args.cfg_t_min < args.cfg_t_max):
        raise ValueError("--cfg-t-min must be smaller than --cfg-t-max when using CFG.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent
    _add_sys_path(project_root / "vendor" / "rae_src")
    _add_sys_path(project_root / "src")

    from stage1 import RAE
    from stage2.transport import Sampler, create_transport
    from utils.model_utils import instantiate_from_config
    from utils.train_utils import parse_configs
    from vae_rae.conditioning import VaeSigregConditioner

    ckpt_path = Path(args.ckpt)
    ckpt = _load_ckpt(ckpt_path)
    if args.config:
        full_cfg = OmegaConf.load(args.config)
    elif "config" in ckpt:
        full_cfg = OmegaConf.create(ckpt["config"])
    else:
        raise ValueError("Checkpoint has no embedded config; pass --config.")

    rae_config, model_config, transport_config, *_ = parse_configs(full_cfg)
    misc_cfg = dict(OmegaConf.to_container(full_cfg.get("misc", {}), resolve=True))

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval().requires_grad_(False)

    vae_cond_cfg = full_cfg.get("vae_condition", None)
    if vae_cond_cfg is None:
        raise ValueError("Config must include `vae_condition`.")
    s1_params = rae_config.get("params", {})
    encoder_params = s1_params.get("encoder_params", {})
    conditioner = VaeSigregConditioner(
        encoder_config_path=str(vae_cond_cfg.get("encoder_config_path", s1_params.get("encoder_config_path"))),
        dinov2_path=str(vae_cond_cfg.get("dinov2_path", encoder_params.get("dinov2_path", s1_params.get("encoder_config_path")))),
        encoder_input_size=int(vae_cond_cfg.get("encoder_input_size", s1_params.get("encoder_input_size", 224))),
        vae_ckpt_path=str(vae_cond_cfg.get("vae_ckpt")),
        vae_src_path=str(vae_cond_cfg.get("vae_src_path", str(project_root.parent / "VAE-SIGREG-SAE" / "src"))),
        sample_temperature=float(vae_cond_cfg.get("sample_temperature", 1.0)),
    ).to(device)
    conditioner.eval().requires_grad_(False)

    if "params" not in model_config:
        model_config["params"] = {}
    model_config["params"]["cond_dim"] = conditioner.cond_dim
    model_config_clean = OmegaConf.create(OmegaConf.to_container(model_config, resolve=True))
    model_config_clean.pop("ckpt", None)
    model = instantiate_from_config(model_config_clean).to(device)
    state_dict = ckpt.get("ema", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[info] missing keys: {missing}", flush=True)
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}", flush=True)
    model.eval().requires_grad_(False)
    del ckpt, state_dict
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[info] loaded checkpoint: {ckpt_path}", flush=True)
    print(f"[info] cond_dim={conditioner.cond_dim}", flush=True)

    transport_params = dict(transport_config.get("params", {}))
    shift_dim = int(misc_cfg.get("time_dist_shift_dim", 768 * 16 * 16))
    shift_base = int(misc_cfg.get("time_dist_shift_base", 4096))
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=math.sqrt(shift_dim / shift_base))
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(sampling_method=args.sampler_method, num_steps=args.num_steps, atol=1e-6, rtol=1e-3)
    latent_size = tuple(int(v) for v in misc_cfg.get("latent_size", [768, 16, 16]))

    cond_path = args.cond_path or args.real_path
    transform = transforms.Compose([CenterCropTransform(args.image_size), transforms.ToTensor()])
    cond_dataset = UnlabeledImageDataset(cond_path, transform=transform, return_path=True)
    args.num_samples = min(args.num_samples, len(cond_dataset))
    print(f"[info] num_samples capped to dataset size: {args.num_samples}", flush=True)
    cond_loader = DataLoader(
        cond_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "cfg_fid"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / f"{ckpt_path.stem}_cfg_results.csv"

    use_amp = args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    results: list[dict[str, float | str | int]] = []

    def cfg_forward(x, t, cfg_scale, cfg_interval=(0.0, 1.0), cond=None):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if cond is None:
            raise ValueError("cond must be provided for CFG.")
        cond_half = cond[: cond.shape[0] // 2] if cond.shape[0] == x.shape[0] else cond
        null_cond = model.null_cond.expand(cond_half.shape[0], -1).to(device=cond_half.device, dtype=cond_half.dtype)
        cond_combined = torch.cat([cond_half, null_cond], dim=0)
        model_out = model.forward(combined, t, cond=cond_combined)
        eps, rest = model_out[:, : model.in_channels], model_out[:, model.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        t_half = t[: len(t) // 2]
        guided_eps = torch.where(
            ((t_half >= cfg_interval[0]) & (t_half <= cfg_interval[1])).view(-1, *[1] * (len(cond_eps.shape) - 1)),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps),
            cond_eps,
        )
        return torch.cat([torch.cat([guided_eps, guided_eps], dim=0), rest], dim=1)

    for cfg_scale in args.cfg_scales:
        scale_name = f"cfg{cfg_scale:g}".replace(".", "p")
        sample_dir = out_dir / ckpt_path.stem / scale_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        pair_grid_dir = out_dir / "pair_grids" / ckpt_path.stem / scale_name
        if args.save_pair_grid:
            pair_grid_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifests" / f"{ckpt_path.stem}_{scale_name}.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        existing = len(list(sample_dir.glob("*.png"))) if args.resume else 0
        if existing:
            print(f"[info] {scale_name}: resume from {existing} samples", flush=True)
        generated = existing
        img_idx = existing
        grid_idx = 0
        grid_saved = 0
        rows: list[tuple[int, str, str]] = []
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        with torch.no_grad():
            while generated < args.num_samples:
                for images_cond, cond_paths in cond_loader:
                    if generated >= args.num_samples:
                        break
                    images_cond = images_cond.to(device, non_blocking=True)
                    bsz = min(images_cond.size(0), args.num_samples - generated)
                    images_cond = images_cond[:bsz]
                    cond_paths = list(cond_paths)[:bsz]
                    cond = conditioner(images_cond).float()
                    if args.add_null:
                        cond = cond + model.null_cond.expand(bsz, -1).to(device=cond.device, dtype=cond.dtype)
                    z0 = torch.randn(bsz, *latent_size, device=device, dtype=torch.float32)
                    if cfg_scale > 1.0:
                        z = torch.cat([z0, z0], dim=0)
                        model_fn = cfg_forward
                        model_kwargs = dict(cond=cond, cfg_scale=cfg_scale, cfg_interval=(args.cfg_t_min, args.cfg_t_max))
                    else:
                        z = z0
                        model_fn = model.forward
                        model_kwargs = dict(cond=cond)

                    with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        zhat = sample_fn(z, model_fn, **model_kwargs)[-1]
                        if cfg_scale > 1.0:
                            zhat = zhat[:bsz]
                        samples = rae.decode(zhat.float()).clamp(0, 1)

                    for i in range(samples.size(0)):
                        out_path = sample_dir / f"{img_idx:06d}.png"
                        save_image(samples[i], out_path, normalize=False)
                        rows.append((img_idx, cond_paths[i], str(out_path)))
                        img_idx += 1
                        generated += 1

                    if args.save_pair_grid and grid_saved < args.max_grid_pairs:
                        n_pairs = min(bsz, args.max_grid_pairs - grid_saved)
                        pair_tensors = []
                        for i in range(n_pairs):
                            pair_tensors.append(images_cond[i].detach().cpu())
                            pair_tensors.append(samples[i].detach().cpu())
                        grid = make_grid(torch.stack(pair_tensors), nrow=2, normalize=False)
                        save_image(grid, pair_grid_dir / f"pair_grid_{grid_idx:04d}.png", normalize=False)
                        grid_idx += 1
                        grid_saved += n_pairs

                    print(f"[info] {scale_name}: {generated}/{args.num_samples} generated", flush=True)

        with manifest_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "condition_input_path", "generated_image_path"])
            writer.writerows(rows)

        print(f"[info] {scale_name}: computing FID", flush=True)
        metrics = calculate_metrics(
            input1=str(sample_dir),
            input2=args.real_path,
            fid=True,
            cuda=True,
            batch_size=min(64, args.batch_size),
            samples_find_deep=True,
            isc=False,
            kid=False,
            prc=False,
        )
        fid_value = float(metrics["frechet_inception_distance"])

        cond_ref_dir = out_dir / "_cond_ref" / f"{ckpt_path.stem}_{scale_name}"
        cond_ref_dir.mkdir(parents=True, exist_ok=True)
        cond_idx = 0
        for images_cond, _ in cond_loader:
            for i in range(images_cond.size(0)):
                if cond_idx >= img_idx:
                    break
                save_image(images_cond[i], cond_ref_dir / f"{cond_idx:06d}.png", normalize=False)
                cond_idx += 1
            if cond_idx >= img_idx:
                break

        paired_metrics = calculate_metrics(
            input1=str(sample_dir),
            input2=str(cond_ref_dir),
            fid=True,
            cuda=True,
            batch_size=min(64, args.batch_size),
            samples_find_deep=False,
            isc=False,
            kid=False,
            prc=False,
        )
        paired_fid_value = float(paired_metrics["frechet_inception_distance"])
        shutil.rmtree(cond_ref_dir, ignore_errors=True)

        print(
            f"[result] {scale_name}: FID={fid_value:.4f}, paired_FID={paired_fid_value:.4f}, samples={img_idx}",
            flush=True,
        )
        results.append(
            {
                "ckpt": str(ckpt_path),
                "cfg_scale": cfg_scale,
                "cfg_t_min": args.cfg_t_min,
                "cfg_t_max": args.cfg_t_max,
                "num_samples": img_idx,
                "fid": fid_value,
                "paired_fid": paired_fid_value,
                "sample_dir": str(sample_dir),
            }
        )

        if not args.keep_samples:
            shutil.rmtree(sample_dir, ignore_errors=True)

    with results_csv.open("w", newline="") as f:
        fieldnames = ["ckpt", "cfg_scale", "cfg_t_min", "cfg_t_max", "num_samples", "fid", "paired_fid", "sample_dir"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[result] wrote summary: {results_csv}", flush=True)


if __name__ == "__main__":
    main()
