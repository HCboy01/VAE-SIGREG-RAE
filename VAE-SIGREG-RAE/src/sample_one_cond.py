#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import amp
from torchvision import transforms
from torchvision.utils import save_image


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample multiple reconstructions from one conditioning image.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--image", type=str, default="")
    p.add_argument("--image-dir", type=str, default="/scratch/x3411a10/datasets/ffhq256/imagefolder/val/images")
    p.add_argument("--image-index", type=int, default=0)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--cfg-t-min", type=float, default=0.0)
    p.add_argument("--cfg-t-max", type=float, default=1.0)
    p.add_argument("--sampler-method", type=str, default="euler")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--seed", type=int, default=42)
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
    from stage2.transport import Sampler, create_transport
    from utils.model_utils import instantiate_from_config
    from vae_rae.conditioning import VaeSigregConditioner

    ckpt = _load_ckpt(args.ckpt)
    cfg = ckpt["config"]

    rae: RAE = instantiate_from_config(OmegaConf.create(cfg["stage_1"])).to(device)
    rae.eval().requires_grad_(False)

    vc = cfg["vae_condition"]
    conditioner = VaeSigregConditioner(
        encoder_config_path=vc["encoder_config_path"],
        dinov2_path=vc["dinov2_path"],
        encoder_input_size=int(vc["encoder_input_size"]),
        vae_ckpt_path=vc["vae_ckpt"],
        vae_src_path=vc["vae_src_path"],
        sample_temperature=float(vc.get("sample_temperature", 1.0)),
    ).to(device)
    conditioner.eval().requires_grad_(False)

    model_cfg = OmegaConf.create(cfg["stage_2"])
    if "ckpt" in model_cfg:
        model_cfg = OmegaConf.create(dict(model_cfg))
        model_cfg.pop("ckpt", None)
    if "params" not in model_cfg:
        model_cfg["params"] = {}
    model_cfg["params"]["cond_dim"] = conditioner.cond_dim
    model = instantiate_from_config(model_cfg).to(device)
    model.load_state_dict(ckpt.get("ema", ckpt.get("model")), strict=True)
    model.eval().requires_grad_(False)

    misc = cfg.get("misc", {})
    shift_dim = int(misc.get("time_dist_shift_dim", 768 * 16 * 16))
    shift_base = int(misc.get("time_dist_shift_base", 4096))
    transport_params = dict(cfg["transport"]["params"])
    transport_params.pop("time_dist_shift", None)
    transport = create_transport(**transport_params, time_dist_shift=math.sqrt(shift_dim / shift_base))
    sample_fn = Sampler(transport).sample_ode(
        sampling_method=args.sampler_method,
        num_steps=args.num_steps,
        atol=1e-6,
        rtol=1e-3,
    )
    latent_size = tuple(int(v) for v in misc.get("latent_size", [768, 16, 16]))

    if args.image:
        image_path = Path(args.image)
    else:
        image_files = sorted(
            p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        image_path = image_files[args.image_index]

    with Image.open(image_path) as img:
        img = img.convert("RGB")
    image = transforms.ToTensor()(CenterCropTransform(args.image_size)(img)).unsqueeze(0).to(device)
    cond = conditioner(image).float()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(image[0].cpu(), out_dir / "condition.png", normalize=False)

    use_amp = args.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    def cfg_forward(x, t, cfg_scale=args.cfg_scale, cfg_interval=(args.cfg_t_min, args.cfg_t_max), cond=None):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        null_cond = model.null_cond.expand(half.shape[0], -1).to(device=cond.device, dtype=cond.dtype)
        cond_combined = torch.cat([cond, null_cond], dim=0)
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

    with torch.no_grad():
        for idx in range(args.num_samples):
            z0 = torch.randn(1, *latent_size, device=device)
            if args.cfg_scale > 1.0:
                z = torch.cat([z0, z0], dim=0)
                with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    sample_z = sample_fn(z, cfg_forward, cond=cond.to(amp_dtype if use_amp else torch.float32))[-1][:1]
            else:
                with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    sample_z = sample_fn(z0, model.forward, cond=cond.to(amp_dtype if use_amp else torch.float32))[-1]
            sample = rae.decode(sample_z.float()).clamp(0, 1)
            save_image(sample[0].cpu(), out_dir / f"sample_{idx:02d}.png", normalize=False)
            print(f"[info] saved sample {idx}: {out_dir / f'sample_{idx:02d}.png'}", flush=True)

    print(f"[result] condition image: {image_path}", flush=True)
    print(f"[result] output dir: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
