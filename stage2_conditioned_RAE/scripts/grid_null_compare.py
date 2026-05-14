#!/usr/bin/env python3
"""Compare regular z_gaussian vs z_gaussian+null_cond unconditional generation.

Top row:    z_gaussian (regular)
Bottom row: z_gaussian + null_cond

Usage:
    cd /scratch/x3411a10/VAE-SIGREG/VAE-SIGREG-RAE
    CUDA_VISIBLE_DEVICES=X python src/grid_null_compare.py \
        --ckpt ckpts/.../best.pt \
        --n-samples 20 \
        --out outputs/null_compare.png
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",          required=True)
    p.add_argument("--n-samples",     type=int, default=20)
    p.add_argument("--cfg-scale",     type=float, default=1.0)
    p.add_argument("--num-steps",     type=int, default=50)
    p.add_argument("--precision",     type=str, default="bf16")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--out",           type=str, default="outputs/null_compare.png")
    return p.parse_args()


def generate(model, rae, sample_fn, z_noise, z_cond, cfg_scale, latent_size, device, use_amp, amp_dtype):
    null_cond = model.null_cond.expand(1, -1).to(device=z_cond.device, dtype=z_cond.dtype)
    cfg_interval = (0.0, 1.0)

    def cfg_fwd(x, t, cond=None):
        half = x[:len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        out = model.forward(combined, t, cond=cond)
        eps, rest = out[:, :model.in_channels], out[:, model.in_channels:]
        c_eps, u_eps = eps[:1], eps[1:]
        t_h = t[:1]
        mask = ((t_h >= cfg_interval[0]) & (t_h <= cfg_interval[1])).view(-1, *[1]*(c_eps.dim()-1))
        guided = torch.where(mask, u_eps + cfg_scale * (c_eps - u_eps), c_eps)
        return torch.cat([torch.cat([guided, guided], dim=0), rest], dim=1)

    z = torch.cat([z_noise, z_noise], dim=0)
    cond_combined = torch.cat([z_cond, null_cond], dim=0)
    with torch.no_grad():
        with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            zhat = sample_fn(z, cfg_fwd, cond=cond_combined)[-1][:1]
            img = rae.decode(zhat.float()).clamp(0, 1)[0].cpu()
    return img


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model = instantiate_from_config(model_cfg).to(device)
    model.load_state_dict(ckpt.get("ema", ckpt.get("model")), strict=True)
    model.eval().requires_grad_(False)

    cond_dim = int(model_cfg.get("params", {}).get("cond_dim", 0))
    print(f"[info] cond_dim={cond_dim}", flush=True)

    misc = cfg.get("misc", {})
    shift_dim  = int(misc.get("time_dist_shift_dim", 768 * 16 * 16))
    shift_base = int(misc.get("time_dist_shift_base", 4096))
    transport_params = dict(cfg["transport"]["params"])
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=math.sqrt(shift_dim / shift_base))
    sample_fn = Sampler(transport).sample_ode(
        sampling_method="euler", num_steps=args.num_steps, atol=1e-6, rtol=1e-3)
    latent_size = tuple(int(v) for v in misc.get("latent_size", [768, 16, 16]))

    use_amp   = args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    null_cond_vec = model.null_cond.to(device)

    row_regular   = []  # z_gaussian
    row_add_null  = []  # z_gaussian + null_cond
    row_null_only = []  # null_cond only (cfg_scale=0)

    for i in range(args.n_samples):
        torch.manual_seed(args.seed + i)
        z_noise = torch.randn(1, *latent_size, device=device)
        z_cond  = torch.randn(1, cond_dim, device=device)

        print(f"[{i+1}/{args.n_samples}] regular ...", flush=True)
        img_regular = generate(model, rae, sample_fn, z_noise, z_cond,
                               args.cfg_scale, latent_size, device, use_amp, amp_dtype)

        z_cond_shifted = z_cond + null_cond_vec.to(dtype=z_cond.dtype)
        print(f"[{i+1}/{args.n_samples}] +null_cond ...", flush=True)
        img_add_null = generate(model, rae, sample_fn, z_noise, z_cond_shifted,
                                args.cfg_scale, latent_size, device, use_amp, amp_dtype)

        print(f"[{i+1}/{args.n_samples}] null only (cfg=0) ...", flush=True)
        img_null_only = generate(model, rae, sample_fn, z_noise, z_cond,
                                 0.0, latent_size, device, use_amp, amp_dtype)

        row_regular.append(img_regular)
        row_add_null.append(img_add_null)
        row_null_only.append(img_null_only)

    row_regular[0]   = add_label(row_regular[0],   "z_gaussian")
    row_add_null[0]  = add_label(row_add_null[0],  "z + null_cond")
    row_null_only[0] = add_label(row_null_only[0], "null only")

    tiles = row_regular + row_add_null + row_null_only
    grid = make_grid(torch.stack(tiles), nrow=args.n_samples, padding=4, pad_value=1.0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(grid).save(out_path)
    print(f"[done] saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
