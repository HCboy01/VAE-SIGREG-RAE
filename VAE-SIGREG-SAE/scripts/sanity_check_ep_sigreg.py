#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vae_sigreg import EppsPulleySIGReg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity check Epps-Pulley SIGReg behavior.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--num-slices", type=int, default=256)
    p.add_argument("--num-points", type=int, default=33)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = EppsPulleySIGReg(
        num_slices=args.num_slices,
        num_points=args.num_points,
        slice_chunk_size=128,
    )

    gaussian = torch.randn(args.batch_size, args.dim, device=device, requires_grad=True)
    shifted = torch.randn(args.batch_size, args.dim, device=device) + 1.5
    high_var = torch.randn(args.batch_size, args.dim, device=device) * 2.0
    uniform = torch.empty(args.batch_size, args.dim, device=device).uniform_(-2.0, 2.0)
    mix_mask = torch.rand(args.batch_size, 1, device=device) > 0.5
    mixture = torch.randn(args.batch_size, args.dim, device=device) + torch.where(mix_mask, 2.0, -2.0)

    losses = {
        "gaussian": loss_fn(gaussian),
        "shifted": loss_fn(shifted),
        "high_var": loss_fn(high_var),
        "uniform": loss_fn(uniform),
        "mixture": loss_fn(mixture),
    }

    losses["gaussian"].backward()
    grad_ok = torch.isfinite(gaussian.grad).all().item()

    for name, value in losses.items():
        print(f"{name:>8}: {value.item():.6f}")
    print(f"backward_grad_finite: {grad_ok}")

    if not grad_ok:
        raise SystemExit("EP-SIGReg backward produced non-finite gradients")
    if not (losses["shifted"] > losses["gaussian"] and losses["high_var"] > losses["gaussian"]):
        raise SystemExit("Expected shifted/high_var losses to exceed Gaussian loss")
    if not (losses["uniform"] > losses["gaussian"] and losses["mixture"] > losses["gaussian"]):
        raise SystemExit("Expected non-Gaussian losses to exceed Gaussian loss")


if __name__ == "__main__":
    main()
