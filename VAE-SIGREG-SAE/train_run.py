"""
Main training script for OvercompleteVariationalAE with SIGReg.

Usage:
    python train_run.py --synthetic                      # smoke test
    python train_run.py --data_path /path/to/embeds.npy  # real data
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

_ENV_PATH = Path("/root/workspace/.env")
if _ENV_PATH.exists():
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import wandb

from src.vae_sigreg import (
    OvercompleteVariationalAE,
    build_sigreg_loss,
    compute_latent_diagnostics,
    kl_bottleneck_loss,
    reconstruction_loss,
    sample_from_prior,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_embeddings(data_path: str) -> torch.Tensor:
    import numpy as np
    p = Path(data_path)
    if p.suffix == ".npy":
        return torch.from_numpy(np.load(str(p))).float()
    if p.suffix in (".pt", ".pth"):
        obj = torch.load(str(p), map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.float()
        if isinstance(obj, dict):
            return next(iter(obj.values())).float()
    if p.suffix == ".bin":
        # binary float32 dump; shape stored in sibling *_shape.npy
        shape_path = p.with_name(p.stem + "_shape.npy")
        if not shape_path.exists():
            raise FileNotFoundError(f"Shape file not found: {shape_path}")
        shape = np.load(str(shape_path))
        return torch.from_numpy(
            np.fromfile(str(p), dtype=np.float32).reshape(shape)
        ).float()
    raise ValueError(f"Unsupported format: {p.suffix}")


def make_dataloaders(embeddings, val_fraction, batch_size, num_workers,
                     val_embeddings=None):
    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=(num_workers > 0))
    if val_embeddings is not None:
        train_ds = TensorDataset(embeddings)
        val_ds   = TensorDataset(val_embeddings)
    else:
        n_val = max(1, int(len(embeddings) * val_fraction))
        n_train = len(embeddings) - n_val
        train_ds, val_ds = random_split(
            TensorDataset(embeddings), [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
    )


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def make_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-6, epoch / warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def latent_batch_metrics(mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor) -> dict:
    """Small aggregate-posterior health metrics for logging."""
    z_f = z.float()
    mu_f = mu.float()
    logvar_f = logvar.float()
    B, D = z_f.shape

    z_mean_abs = z_f.mean(dim=0).abs().mean()
    z_var_err = (z_f.var(dim=0, unbiased=False) - 1).abs().mean()

    n_sub = min(64, D)
    z_sub = z_f[:, :n_sub]
    z_sub_c = z_sub - z_sub.mean(dim=0)
    cov = z_sub_c.T @ z_sub_c / max(B - 1, 1)
    off_mask = ~torch.eye(n_sub, dtype=torch.bool, device=z.device)
    off_diag_cov = cov[off_mask].abs().mean()

    kl_dim = 0.5 * (mu_f.pow(2) + logvar_f.exp() - logvar_f - 1).mean(dim=0)
    active_units = (kl_dim > 0.01).sum()

    return {
        "latent_mean_error": z_mean_abs.item(),
        "latent_variance_error": z_var_err.item(),
        "covariance_offdiag_error": off_diag_cov.item(),
        "active_units": active_units.item(),
    }


def threshold_tag(value: float) -> str:
    """Make a stable metric suffix for activation thresholds."""
    return f"{value:g}".replace("-", "m").replace(".", "_")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_path",     type=str,   default=None)
    p.add_argument("--val_data_path", type=str,   default=None,
                   help="separate val set; if omitted, split from data_path")
    p.add_argument("--synthetic",     action="store_true")
    p.add_argument("--n_synthetic",   type=int,   default=100_000)
    p.add_argument("--val_fraction",  type=float, default=0.05)
    p.add_argument("--num_workers",   type=int,   default=4)

    # model
    p.add_argument("--input_dim",  type=int,   default=768)
    p.add_argument("--latent_dim", type=int,   default=6144)
    p.add_argument("--hidden_dim", type=int,   default=None)
    p.add_argument("--num_layers", type=int,   default=4)

    # training
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=512)
    p.add_argument("--accum_steps", type=int,   default=4,   help="gradient accumulation (effective batch = batch*accum)")
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--weight_decay",type=float, default=1e-2)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--mixed_precision", action="store_true", default=True)
    p.add_argument("--no_mixed_precision", dest="mixed_precision", action="store_false")

    # LR schedule
    p.add_argument("--warmup_epochs", type=int, default=5)

    # loss weights
    p.add_argument("--beta_kl",             type=float, default=1e-3)
    p.add_argument("--beta_kl_warmup_epochs",type=int,  default=30,  help="linearly ramp beta_kl from 0 over N epochs")
    p.add_argument("--lambda_sigreg",       type=float, default=0.1)
    p.add_argument("--num_projections",     type=int,   default=512)
    p.add_argument("--sigreg_type", type=str, default="epps_pulley",
                   choices=["epps_pulley", "moment", "none"])
    p.add_argument("--sigreg_target", type=str, default="z", choices=["z", "mu"],
                   help="Apply SIGReg to stochastic z or deterministic mu")
    p.add_argument("--ep_num_points", type=int, default=33)
    p.add_argument("--ep_t_min", type=float, default=-5.0)
    p.add_argument("--ep_t_max", type=float, default=5.0)
    p.add_argument("--ep_no_weight", action="store_true",
                   help="Disable exp(-0.5*t^2) weighting in EP-SIGReg")
    p.add_argument("--ep_slice_chunk_size", type=int, default=256)

    # checkpointing: keep only the best validation checkpoint to avoid quota blowups.
    p.add_argument("--ckpt_dir",   type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--best_metric", type=str, default="total_loss",
                   choices=["total_loss", "active_safe_total", "active_units"],
                   help="Checkpoint selection. active_safe_total keeps the best total among epochs above --min_active_units.")
    p.add_argument("--min_active_units", type=float, default=0.0,
                   help="Minimum val/active_units required when --best_metric=active_safe_total.")
    p.add_argument("--dead_activation_threshold", type=float, default=0.1,
                   help="A latent dimension is dead if it never exceeds this absolute activation over the validation set.")
    p.add_argument("--dead_activation_thresholds", type=float, nargs="*",
                   default=[0.1, 0.5],
                   help="Thresholds logged for dead-neuron counting over the full validation set.")
    p.add_argument("--dead_activation_target", type=str, default="mu", choices=["mu", "z"],
                   help="Target used for dead-neuron counting. mu is deterministic; z includes sampling noise.")
    p.add_argument("--resume",       type=str, default=None, help="full resume: load model+optimizer+epoch")
    p.add_argument("--init_weights", type=str, default=None, help="load model weights only, train from epoch 1")

    # wandb
    p.add_argument("--wandb_project",  type=str,  default="vae-sigreg-sae")
    p.add_argument("--wandb_run_name", type=str,  default=None)
    p.add_argument("--wandb_tags",     type=str,  nargs="*", default=[])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.mixed_precision and device.type == "cuda"
    amp_dtype = torch.bfloat16  # bf16: stable without GradScaler
    print(f"Device: {device}  |  AMP: {use_amp} ({amp_dtype})")

    # --- Data ---
    if args.synthetic:
        print(f"Synthetic data: {args.n_synthetic} x {args.input_dim}")
        embeddings = torch.randn(args.n_synthetic, args.input_dim)
        val_embeddings = None
    elif args.data_path:
        print(f"Loading train: {args.data_path}")
        embeddings = load_embeddings(args.data_path)
        print(f"  train shape={embeddings.shape}")
        val_embeddings = None
        if args.val_data_path:
            print(f"Loading val:   {args.val_data_path}")
            val_embeddings = load_embeddings(args.val_data_path)
            print(f"  val   shape={val_embeddings.shape}")
    else:
        sys.exit("Provide --data_path or --synthetic")

    train_loader, val_loader = make_dataloaders(
        embeddings, args.val_fraction, args.batch_size, args.num_workers,
        val_embeddings=val_embeddings,
    )
    eff_batch = args.batch_size * args.accum_steps
    print(f"Train={len(train_loader)} batches  Val={len(val_loader)} batches  "
          f"eff_batch={eff_batch}")

    # --- Model ---
    model = OvercompleteVariationalAE(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),  # more aggressive momentum for large batch
    )
    scheduler = make_scheduler(optimizer, args.warmup_epochs, args.epochs)
    sigreg_module = build_sigreg_loss(
        sigreg_type=args.sigreg_type,
        num_projections=args.num_projections,
        ep_num_points=args.ep_num_points,
        ep_t_min=args.ep_t_min,
        ep_t_max=args.ep_t_max,
        ep_weighted=not args.ep_no_weight,
        ep_slice_chunk_size=args.ep_slice_chunk_size,
    )

    # --- Resume / Init ---
    start_epoch = 1
    best_val_total = float("inf")
    best_active_units = -float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"Resumed from {args.resume}  (epoch {ckpt['epoch']} → continuing from {start_epoch})")
    elif args.init_weights:
        ckpt = torch.load(args.init_weights, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Weights loaded from {args.init_weights}  (training from epoch 1)")

    # --- wandb ---
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        tags=args.wandb_tags,
        config={**vars(args), "eff_batch": eff_batch, "n_params": n_params},
    )
    wandb.watch(model, log=None)  # gradient logging disabled: too slow with 90M params

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # KL annealing: linearly ramp beta from 0 → beta_kl over warmup epochs
        beta_kl_now = args.beta_kl * min(1.0, epoch / max(1, args.beta_kl_warmup_epochs))

        # ---- Train ----
        model.train()
        train_acc = {}
        n_train = 0
        optimizer.zero_grad()

        for step, (x,) in enumerate(train_loader):
            x = x.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(x)
                rec  = reconstruction_loss(out["x_hat"], x)
                kl   = kl_bottleneck_loss(out["mu"], out["logvar"])
                sig_target = out[args.sigreg_target]
                sig  = sigreg_module(sig_target)
                loss = (rec + beta_kl_now * kl + args.lambda_sigreg * sig) / args.accum_steps

            loss.backward()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            # accumulate (unscaled) metrics
            s = args.accum_steps
            batch_health = latent_batch_metrics(out["mu"], out["logvar"], out["z"])
            for k, v in [
                ("rec_loss", rec), ("kl_loss", kl), ("sigreg_loss", sig),
                ("ep_sigreg_loss", sig if args.sigreg_type == "epps_pulley" else sig.detach() * 0.0),
                ("total_loss", rec + beta_kl_now * kl + args.lambda_sigreg * sig),
                ("mu_sq_mean", out["mu"].pow(2).mean().detach()),
                ("exp_logvar_mean", out["logvar"].exp().mean().detach()),
            ]:
                train_acc[k] = train_acc.get(k, 0.0) + v.item()
            for k, v in batch_health.items():
                train_acc[k] = train_acc.get(k, 0.0) + float(v)
            n_train += 1

        train_metrics = {k: v / n_train for k, v in train_acc.items()}
        scheduler.step()

        # ---- Eval ----
        model.eval()
        val_acc = {}
        n_val = 0
        dead_thresholds = sorted(set(args.dead_activation_thresholds + [args.dead_activation_threshold]))
        ever_active_masks = {thr: None for thr in dead_thresholds}
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(x)
                    rec = reconstruction_loss(out["x_hat"], x)
                    kl  = kl_bottleneck_loss(out["mu"], out["logvar"])
                    sig = sigreg_module(out[args.sigreg_target])

                batch_health = latent_batch_metrics(out["mu"], out["logvar"], out["z"])
                for k, v in [
                    ("rec_loss", rec), ("kl_loss", kl), ("sigreg_loss", sig),
                    ("ep_sigreg_loss", sig if args.sigreg_type == "epps_pulley" else sig * 0.0),
                    ("total_loss", rec + beta_kl_now * kl + args.lambda_sigreg * sig),
                    ("mu_sq_mean", out["mu"].pow(2).mean()),
                    ("exp_logvar_mean", out["logvar"].exp().mean()),
                ]:
                    val_acc[k] = val_acc.get(k, 0.0) + v.item()
                for k, v in batch_health.items():
                    val_acc[k] = val_acc.get(k, 0.0) + float(v)
                n_val += 1

                dead_target = out[args.dead_activation_target].float()
                dead_target_abs = dead_target.abs()
                for thr in dead_thresholds:
                    batch_active = dead_target_abs.gt(thr).any(dim=0)
                    prev = ever_active_masks[thr]
                    ever_active_masks[thr] = batch_active if prev is None else (prev | batch_active)

        val_metrics = {k: v / n_val for k, v in val_acc.items()}
        primary_dead_metrics = None
        for thr, mask in ever_active_masks.items():
            if mask is None:
                continue
            ever_active_units = mask.sum().item()
            dead_units = mask.numel() - ever_active_units
            tag = threshold_tag(thr)
            metrics = {
                "ever_active_units": float(ever_active_units),
                "dead_units": float(dead_units),
                "dead_fraction": float(dead_units / max(1, mask.numel())),
            }
            val_metrics[f"ever_active_units_thr_{tag}"] = metrics["ever_active_units"]
            val_metrics[f"dead_units_thr_{tag}"] = metrics["dead_units"]
            val_metrics[f"dead_fraction_thr_{tag}"] = metrics["dead_fraction"]
            if abs(thr - args.dead_activation_threshold) < 1e-12:
                primary_dead_metrics = metrics
        if primary_dead_metrics is not None:
            val_metrics["ever_active_units"] = primary_dead_metrics["ever_active_units"]
            val_metrics["dead_units"] = primary_dead_metrics["dead_units"]
            val_metrics["dead_fraction"] = primary_dead_metrics["dead_fraction"]
            val_metrics["active_units_kl_0_01"] = val_metrics.get("active_units", 0.0)

        # ---- Latent diagnostics (one val batch) ----
        with torch.no_grad():
            x_diag = next(iter(val_loader))[0].to(device)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                z_diag = model(x_diag)["z"].float()
        diag = compute_latent_diagnostics(z_diag)

        # also log z stats inline
        with torch.no_grad():
            z_mean_abs = z_diag.mean(dim=0).abs().mean().item()
            z_var_err  = (z_diag.var(dim=0) - 1).abs().mean().item()
            n_sub = min(64, z_diag.shape[1])
            z_sub = z_diag[:, :n_sub]
            z_sub_c = z_sub - z_sub.mean(dim=0)
            cov = z_sub_c.T @ z_sub_c / max(z_sub.shape[0] - 1, 1)
            off_mask = ~torch.eye(n_sub, dtype=torch.bool, device=device)
            off_diag_cov = cov[off_mask].abs().mean().item()

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        # ---- wandb log ----
        log = {
            "epoch": epoch,
            "lr": lr_now,
            "beta_kl_now": beta_kl_now,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"val/{k}": v for k, v in val_metrics.items()},
            "val/z_mean_abs": z_mean_abs,
            "val/z_var_err": z_var_err,
            "val/z_off_diag_cov": off_diag_cov,
            "val/latent_mean_error": val_metrics.get("latent_mean_error", z_mean_abs),
            "val/latent_variance_error": val_metrics.get("latent_variance_error", z_var_err),
            "val/covariance_offdiag_error": val_metrics.get("covariance_offdiag_error", off_diag_cov),
            "val/active_units": val_metrics.get("active_units", 0.0),
            "val/ever_active_units": val_metrics.get("ever_active_units", 0.0),
            "val/dead_units": val_metrics.get("dead_units", 0.0),
            "val/dead_fraction": val_metrics.get("dead_fraction", 0.0),
            "val/active_units_kl_0_01": val_metrics.get("active_units_kl_0_01", val_metrics.get("active_units", 0.0)),
            **{f"diag/{k}": v for k, v in diag.items()},
            "epoch_time_sec": elapsed,
        }
        wandb.log(log, step=epoch)

        print(
            f"[{epoch:04d}/{args.epochs}] "
            f"lr={lr_now:.2e}  β={beta_kl_now:.2e}  "
            f"rec={val_metrics['rec_loss']:.4f}  "
            f"kl={val_metrics['kl_loss']:.1f}  "
            f"sig={val_metrics['sigreg_loss']:.4f}  "
            f"total={val_metrics['total_loss']:.4f}  "
            f"ever_active={val_metrics.get('ever_active_units', 0.0):.1f}  "
            f"dead={val_metrics.get('dead_units', 0.0):.1f}  "
            f"kl_active={val_metrics.get('active_units', 0.0):.1f}  "
            f"var={val_metrics.get('latent_variance_error', z_var_err):.4f}  "
            f"offdiag={val_metrics.get('covariance_offdiag_error', off_diag_cov):.4f}  "
            f"({elapsed:.1f}s)"
        )

        # ---- Checkpoint ----
        active_now = val_metrics.get("ever_active_units", val_metrics.get("active_units", 0.0))
        if args.best_metric == "active_units":
            is_best = active_now > best_active_units
        elif args.best_metric == "active_safe_total":
            is_best = active_now >= args.min_active_units and val_metrics["total_loss"] < best_val_total
        else:
            is_best = val_metrics["total_loss"] < best_val_total
        if is_best:
            best_val_total = val_metrics["total_loss"]
            best_active_units = active_now
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "args": vars(args)},
                ckpt_dir / "best.pt",
            )

    samples = sample_from_prior(model, 8, args.latent_dim, device)
    print(f"Prior samples: {samples.shape}")
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
