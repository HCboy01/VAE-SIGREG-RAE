#!/usr/bin/env python3
"""FID evaluation with unconditional Gaussian random sampling as condition.

Replaces the VAE-SIGREG conditioner with z_cond ~ N(0, I) and generates
`num-samples` images, then computes FID against a real image directory.

Usage:
    python src/eval_gaussian_fid.py \
        --ckpt ckpts/vae_sigreg_mu_cond/ep-0040.pt \
        --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_VAESIGREG.yaml \
        --real-path /scratch/x3411a10/datasets/ffhq256/imagefolder/val \
        --num-samples 5000 \
        --batch-size 16 \
        --precision bf16
"""
from __future__ import annotations

import argparse
import gc
import inspect
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import amp
from torchvision.utils import save_image
from torch_fidelity import calculate_metrics


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FID eval with Gaussian random condition")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--real-path", type=str, required=True, help="Real image directory for FID reference")
    p.add_argument("--out-dir", type=str, default="", help="Where to save generated images (default: <ckpt_dir>/gaussian_fid)")
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--cfg-t-min", type=float, default=0.0)
    p.add_argument("--cfg-t-max", type=float, default=1.0)
    p.add_argument("--sampler-method", type=str, default="euler")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--keep-samples", action="store_true", help="Keep generated images after FID computation")
    p.add_argument("--resume", action="store_true", help="Skip already-generated images in out-dir and continue")
    p.add_argument("--add-null", action="store_true", help="Add null_cond to z_gaussian before conditioning")
    p.add_argument("--az-scale", type=float, default=None, help="Use null_cond + az_scale*z_gaussian as condition")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cfg_scale > 1.0 and not (args.cfg_t_min < args.cfg_t_max):
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
    from stage2.transport import create_transport, Sampler
    from utils.model_utils import instantiate_from_config
    from utils.train_utils import parse_configs

    full_cfg = OmegaConf.load(args.config)
    rae_config, model_config, transport_config, _, _, _, _, _ = parse_configs(full_cfg)
    misc_cfg = dict(OmegaConf.to_container(full_cfg.get("misc", {}), resolve=True))

    # ── RAE decoder ──────────────────────────────────────────────────────────
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval().requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()

    # ── cond_dim: read directly from cond_proj.weight in checkpoint ──────────
    raw_ckpt = _load_ckpt(args.ckpt)
    state_dict = raw_ckpt.get("ema", raw_ckpt.get("model", raw_ckpt))
    cond_proj_w = state_dict.get("cond_proj.weight")
    if cond_proj_w is None:
        raise RuntimeError("Could not find 'cond_proj.weight' in checkpoint. Is this a VAE-conditioned model?")
    cond_dim = cond_proj_w.shape[1]
    print(f"[info] cond_dim = {cond_dim}  (from cond_proj.weight shape {tuple(cond_proj_w.shape)})", flush=True)

    # ── DiT model ─────────────────────────────────────────────────────────────
    if "params" not in model_config:
        model_config["params"] = {}
    model_config["params"]["cond_dim"] = cond_dim
    model_config_clean = OmegaConf.create(OmegaConf.to_container(model_config, resolve=True))
    model_config_clean.pop("ckpt", None)

    model = instantiate_from_config(model_config_clean).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[info] missing keys: {missing}", flush=True)
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}", flush=True)
    del raw_ckpt, state_dict, cond_proj_w
    gc.collect()
    model.eval().requires_grad_(False)
    print(f"[info] loaded checkpoint: {args.ckpt}", flush=True)

    # ── Transport / sampler ───────────────────────────────────────────────────
    transport_params = dict(transport_config.get("params", {}))
    shift_dim = int(misc_cfg.get("time_dist_shift_dim", 196608))
    shift_base = int(misc_cfg.get("time_dist_shift_base", 4096))
    time_dist_shift = math.sqrt(shift_dim / shift_base)
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=time_dist_shift)
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(sampling_method=args.sampler_method, num_steps=args.num_steps, atol=1e-6, rtol=1e-3)

    latent_size = tuple(int(v) for v in misc_cfg.get("latent_size", [768, 16, 16]))

    # ── Output dir ────────────────────────────────────────────────────────────
    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "gaussian_fid"
    scale_name = f"cfg{args.cfg_scale:g}".replace(".", "p")
    sample_dir = out_dir / ckpt_path.stem / scale_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] saving generated images to: {sample_dir}", flush=True)

    # ── Generation with z_cond ~ N(0, I) ─────────────────────────────────────
    use_amp = args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    existing = len(list(sample_dir.glob("*.png"))) if args.resume else 0
    if existing > 0:
        print(f"[info] resume: skipping {existing} already-generated images", flush=True)
    generated = existing
    img_idx = existing
    # advance RNG to stay consistent with original run
    if existing > 0:
        torch.manual_seed(args.seed + existing)
        torch.cuda.manual_seed_all(args.seed + existing)
    with torch.no_grad():
        while generated < args.num_samples:
            bsz = min(args.batch_size, args.num_samples - generated)

            z_noise = torch.randn(bsz, *latent_size, device=device)
            z_cond = torch.randn(bsz, cond_dim, device=device)
            if args.az_scale is not None:
                null = model.null_cond.expand(bsz, -1).to(device=z_cond.device, dtype=z_cond.dtype)
                z_cond = null + args.az_scale * z_cond
            elif args.add_null:
                z_cond = z_cond + model.null_cond.expand(bsz, -1).to(device=z_cond.device, dtype=z_cond.dtype)
            z = torch.cat([z_noise, z_noise], dim=0)
            cond = z_cond.to(amp_dtype if use_amp else torch.float32)
            null_cond = model.null_cond.expand(bsz, -1).to(device=cond.device, dtype=cond.dtype)
            cond_combined = torch.cat([cond, null_cond], dim=0)
            cfg_scale = args.cfg_scale
            cfg_interval = (args.cfg_t_min, args.cfg_t_max)

            def cfg_forward(x, t, cond=None):
                half = x[: len(x) // 2]
                combined = torch.cat([half, half], dim=0)
                model_out = model.forward(combined, t, cond=cond)
                eps, rest = model_out[:, : model.in_channels], model_out[:, model.in_channels :]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
                t_half = t[: len(t) // 2]
                guided_eps = torch.where(
                    ((t_half >= cfg_interval[0]) & (t_half <= cfg_interval[1])).view(
                        -1, *[1] * (len(cond_eps.shape) - 1)
                    ),
                    uncond_eps + cfg_scale * (cond_eps - uncond_eps),
                    cond_eps,
                )
                return torch.cat([torch.cat([guided_eps, guided_eps], dim=0), rest], dim=1)

            with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                zhat = sample_fn(z, cfg_forward, cond=cond_combined)[-1][:bsz]
            samples = rae.decode(zhat.float()).clamp(0, 1)

            for i in range(samples.size(0)):
                save_image(samples[i], sample_dir / f"{img_idx:06d}.png", normalize=False)
                img_idx += 1

            generated += bsz
            print(f"[info] {generated}/{args.num_samples} generated", flush=True)

    print(f"[info] generation done. computing FID...", flush=True)

    # ── FID ───────────────────────────────────────────────────────────────────
    metrics = calculate_metrics(
        input1=str(sample_dir),
        input2=args.real_path,
        fid=True,
        cuda=(device.type == "cuda"),
        batch_size=min(64, args.batch_size),
        samples_find_deep=True,
        isc=False,
        kid=False,
        prc=False,
    )
    fid_value = float(metrics["frechet_inception_distance"])
    print(f"\n[result] FID (gaussian random cond, cfg={args.cfg_scale:g}, {generated} samples) = {fid_value:.4f}", flush=True)
    print(f"[result] ckpt: {args.ckpt}", flush=True)

    if not args.keep_samples:
        import shutil
        shutil.rmtree(sample_dir)
        print(f"[info] removed generated images (use --keep-samples to retain)", flush=True)


if __name__ == "__main__":
    main()
