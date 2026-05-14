#!/usr/bin/env python3
"""Generate a CFG-scale sweep grid from a single Gaussian random condition.

z_cond ~ N(0, I) (unconditional), same z_noise for all scales.

Usage:
    cd /scratch/x3411a10/VAE-SIGREG/VAE-SIGREG-RAE
    CUDA_VISIBLE_DEVICES=1 python src/grid_cfg_sweep.py \
        --ckpt ckpts/vae_sigreg_betakl3e5_nullfix_40ep_20260503_184127/ep-0020.pt \
        --cfg-scales 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 \
        --out cfg_sweep_grid.png
"""
from __future__ import annotations

import argparse
import inspect
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch import amp
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF


def _add_sys_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_ckpt(path: str | Path) -> dict:
    sig = inspect.signature(torch.load)
    kwargs = {"map_location": "cpu"}
    if "mmap" in sig.parameters:
        kwargs["mmap"] = True
    if "weights_only" in sig.parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",         required=True)
    p.add_argument("--cfg-scales",   type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0])
    p.add_argument("--n-samples",    type=int, default=10)
    p.add_argument("--cfg-t-min",    type=float, default=0.0)
    p.add_argument("--cfg-t-max",    type=float, default=1.0)
    p.add_argument("--sampler-method", type=str, default="euler")
    p.add_argument("--num-steps",    type=int, default=50)
    p.add_argument("--precision",    type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--out",          type=str, default="cfg_sweep_grid.png")
    p.add_argument("--add-null",     action="store_true", help="Add null_cond to z_gaussian before conditioning")
    return p.parse_args()


def add_label(tensor_chw: torch.Tensor, text: str, font_size: int = 18) -> torch.Tensor:
    img = TF.to_pil_image(tensor_chw.clamp(0, 1))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, font_size + 6], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255), font=font)
    return TF.to_tensor(img)


def cfg_forward(model, x, t, cond, cfg_scale, cfg_t_min, cfg_t_max):
    bsz = x.shape[0] // 2
    half = x[:bsz]
    combined = torch.cat([half, half], dim=0)
    null_cond = model.null_cond.expand(bsz, -1).to(device=cond.device, dtype=cond.dtype)
    cond_combined = torch.cat([cond, null_cond], dim=0)
    model_out = model.forward(combined, t, cond=cond_combined)
    eps, rest = model_out[:, :model.in_channels], model_out[:, model.in_channels:]
    cond_eps, uncond_eps = eps[:bsz], eps[bsz:]
    t_half = t[:bsz]
    in_interval = ((t_half >= cfg_t_min) & (t_half <= cfg_t_max)).view(-1, *[1] * (cond_eps.dim() - 1))
    guided = torch.where(in_interval, uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps)
    return torch.cat([torch.cat([guided, guided], dim=0), rest], dim=1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required.")

    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent
    _add_sys_path(project_root / "vendor" / "rae_src")
    _add_sys_path(project_root / "src")

    from stage1 import RAE
    from stage2.transport import Sampler, create_transport
    from utils.model_utils import instantiate_from_config

    ckpt = _load_ckpt(args.ckpt)
    cfg = ckpt["config"]

    print("[info] loading RAE ...", flush=True)
    rae: RAE = instantiate_from_config(OmegaConf.create(cfg["stage_1"])).to(device)
    rae.eval().requires_grad_(False)

    print("[info] loading flow model ...", flush=True)
    model_cfg = OmegaConf.create(dict(cfg["stage_2"]))
    model_cfg.pop("ckpt", None)
    cond_dim = int(model_cfg.get("params", {}).get("cond_dim", 0))
    model = instantiate_from_config(model_cfg).to(device)
    model.load_state_dict(ckpt.get("ema", ckpt.get("model")), strict=True)
    model.eval().requires_grad_(False)
    print(f"[info] cond_dim={cond_dim}", flush=True)

    misc = cfg.get("misc", {})
    shift_dim  = int(misc.get("time_dist_shift_dim", 768 * 16 * 16))
    shift_base = int(misc.get("time_dist_shift_base", 4096))
    transport_params = dict(cfg["transport"]["params"])
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=math.sqrt(shift_dim / shift_base))
    sample_fn = Sampler(transport).sample_ode(
        sampling_method=args.sampler_method, num_steps=args.num_steps, atol=1e-6, rtol=1e-3)
    latent_size = tuple(int(v) for v in misc.get("latent_size", [768, 16, 16]))

    use_amp   = args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    # rows=samples, cols=cfg_scales
    # each row shares the same z_noise and z_cond
    tiles = []  # row-major: [sample0_cfg0, sample0_cfg1, ..., sample1_cfg0, ...]
    for sample_idx in range(args.n_samples):
        torch.manual_seed(args.seed + sample_idx)
        z_noise = torch.randn(1, *latent_size, device=device)
        z_cond  = torch.randn(1, cond_dim, device=device) if cond_dim > 0 else None
        if z_cond is not None and args.add_null:
            z_cond = z_cond + model.null_cond.to(device=z_cond.device, dtype=z_cond.dtype)

        for scale in args.cfg_scales:
            print(f"[info] sample={sample_idx}  CFG={scale:.2f} ...", flush=True)
            with torch.no_grad():
                if z_cond is not None:
                    z = torch.cat([z_noise, z_noise], dim=0)
                    cond_in = z_cond.to(amp_dtype if use_amp else torch.float32)
                    fn = lambda x, t, s=scale: cfg_forward(model, x, t, cond_in, s, args.cfg_t_min, args.cfg_t_max)
                    with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        zhat = sample_fn(z, fn)[-1][:1]
                else:
                    with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        zhat = sample_fn(z_noise, model.forward)[-1]
                img = rae.decode(zhat.float()).clamp(0, 1)[0].cpu()

            label = f"CFG {scale:.2f}" if sample_idx == 0 else ""
            tiles.append(add_label(img, label) if label else img)

    grid = make_grid(torch.stack(tiles), nrow=len(args.cfg_scales), padding=4, pad_value=1.0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(grid).save(out_path)
    print(f"[done] saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
