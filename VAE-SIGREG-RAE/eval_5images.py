#!/usr/bin/env python3
"""
eval dataset 첫 5개 이미지에 대해 VAE-SIGREG-RAE reconstruction 수행.
각 이미지마다 원본 + 1번 재구성 결과를 그리드로 저장.
"""
from __future__ import annotations

import sys
import math
import inspect
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

THIS_DIR = Path(__file__).resolve().parent
RAE_SRC = THIS_DIR / "vendor" / "rae_src"
VAE_RAE_SRC = THIS_DIR / "src"
VAE_SAE_SRC = THIS_DIR.parent / "VAE-SIGREG-SAE" / "src"

for p in [RAE_SRC, VAE_RAE_SRC, VAE_SAE_SRC]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _full_load(path, map_location="cpu"):
    kwargs = {"map_location": map_location}
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


class CenterCropTransform:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        import numpy as np
        s = self.size
        while min(*img.size) >= 2 * s:
            img = img.resize(tuple(x // 2 for x in img.size), resample=Image.BOX)
        scale = s / min(*img.size)
        img = img.resize(tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC)
        arr = np.array(img)
        cy = (arr.shape[0] - s) // 2
        cx = (arr.shape[1] - s) // 2
        return Image.fromarray(arr[cy: cy + s, cx: cx + s])


import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--ckpt", type=str, default="/scratch/x3411a10/VAE-SIGREG/VAE-SIGREG-RAE/ckpts/vae_sigreg_betakl3e5/best.pt")
_parser.add_argument("--output", type=str, default=str(THIS_DIR / "eval_5images_result.png"))
_args = _parser.parse_args()

CKPT_PATH  = Path(_args.ckpt)
VAL_DIR    = Path("/scratch/x3411a10/datasets/ffhq256/imagefolder/val/images")
OUTPUT_PATH = Path(_args.output)
N_IMAGES   = 5
IMAGE_SIZE = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device: {DEVICE}")
print(f"checkpoint: {CKPT_PATH}")

ckpt = _full_load(CKPT_PATH, map_location="cpu")
cfg  = ckpt["config"]

from stage1 import RAE
from stage2.transport import create_transport, Sampler
from utils.model_utils import instantiate_from_config
from vae_rae.conditioning import VaeSigregConditioner
from omegaconf import OmegaConf

rae_cfg = OmegaConf.create(cfg["stage_1"])
rae: RAE = instantiate_from_config(rae_cfg).to(DEVICE)
rae.eval().requires_grad_(False)
print("RAE loaded.")

vc = cfg["vae_condition"]
conditioner = VaeSigregConditioner(
    encoder_config_path=vc["encoder_config_path"],
    dinov2_path=vc["dinov2_path"],
    encoder_input_size=int(vc["encoder_input_size"]),
    vae_ckpt_path=vc["vae_ckpt"],
    vae_src_path=vc["vae_src_path"],
    sample_temperature=float(vc.get("sample_temperature", 1.0)),
).to(DEVICE)
conditioner.eval().requires_grad_(False)
print("Conditioner loaded.")

s2_cfg = OmegaConf.create(cfg["stage_2"])
if "ckpt" in s2_cfg:
    s2_cfg = OmegaConf.create(dict(s2_cfg))
    s2_cfg.pop("ckpt", None)
if "params" not in s2_cfg:
    s2_cfg["params"] = {}
s2_cfg["params"]["cond_dim"] = conditioner.cond_dim

model = instantiate_from_config(s2_cfg).to(DEVICE)
state_dict = ckpt.get("ema", ckpt.get("model"))
model.load_state_dict(state_dict, strict=True)
model.eval().requires_grad_(False)
print("Stage-2 model (EMA) loaded.")

misc = cfg.get("misc", {})
shift_dim  = int(misc.get("time_dist_shift_dim", 768 * 16 * 16))
shift_base = int(misc.get("time_dist_shift_base", 4096))
time_dist_shift = math.sqrt(shift_dim / shift_base)

tp = cfg["transport"]["params"]
transport = create_transport(
    path_type=tp.get("path_type", "Linear"),
    prediction=tp.get("prediction", "velocity"),
    loss_weight=tp.get("loss_weight", None),
    time_dist_type=tp.get("time_dist_type", "logit-normal_0_1"),
    time_dist_shift=time_dist_shift,
)
sampler   = Sampler(transport)
sample_fn = sampler.sample_ode(sampling_method="dopri5", num_steps=50, atol=1e-6, rtol=1e-3)

latent_size = tuple(int(v) for v in misc.get("latent_size", [768, 16, 16]))
print(f"latent_size: {latent_size}")

image_files = sorted(p for p in VAL_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))[:N_IMAGES]
print(f"\n처리할 이미지: {[f.name for f in image_files]}")

crop_tf   = CenterCropTransform(IMAGE_SIZE)
to_tensor = transforms.ToTensor()

all_pairs = []

with torch.no_grad():
    for i, img_path in enumerate(image_files):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        img_tensor = to_tensor(crop_tf(img)).unsqueeze(0).to(DEVICE)

        cond = conditioner(img_tensor)
        z0   = torch.randn(1, *latent_size, device=DEVICE)
        zhat = sample_fn(z0, model.forward, cond=cond)[-1]
        recon = rae.decode(zhat).clamp(0, 1)

        all_pairs.append(img_tensor.cpu())
        all_pairs.append(recon.cpu())
        print(f"  [{i+1}/{N_IMAGES}] {img_path.name} done")

# 2행 N_IMAGES열: 홀수행=원본, 짝수행=재구성
grid = make_grid(torch.cat(all_pairs, dim=0), nrow=N_IMAGES, padding=4, normalize=False)
save_image(grid, OUTPUT_PATH, normalize=False)
print(f"\n결과 저장: {OUTPUT_PATH}")
