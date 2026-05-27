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
    compute_kl_active_dims,
    compute_latent_diagnostics,
    compute_marginal_moments,
    compute_selective_activity,
    compute_z_frechet_distance,
    kl_bottleneck_loss,
    kl_feature_level,
    kl_feature_level_log,
    kl_feature_level_sq,
    l1_mu_loss,
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
    p.add_argument("--linear_decoder", action="store_true", default=True,
                   help="use single linear layer as decoder (default: True)")
    p.add_argument("--no_linear_decoder", dest="linear_decoder", action="store_false",
                   help="use MLP decoder instead of linear")

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
    p.add_argument("--kl_type", type=str, default="sample",
                   choices=["sample", "feature", "feature_sq", "feature_log", "sample_l1"],
                   help="sample: standard per-image KL (default). "
                        "feature: marginal KL per dim — penalises always-on dims directly. "
                        "feature_sq: sum_d(mean_n[KL_d])^2 — no cancellation, gradient proportional to mean KL. "
                        "feature_log: sum_d log(1+mean_n[KL_d]) — sparsity-inducing, gradient saturates for active dims. "
                        "sample_l1: standard KL + alpha_l1 * mean|mu| — Laplace prior sharpens sparsity.")
    p.add_argument("--alpha_l1",            type=float, default=0.0,
                   help="weight for L1 penalty on |mu| (only used when kl_type=sample_l1)")
    p.add_argument("--lambda_sigreg",       type=float, default=0.1)
    p.add_argument("--num_projections",     type=int,   default=512)
    p.add_argument("--sigreg_type", type=str, default="epps_pulley",
                   choices=["epps_pulley", "moment", "none"])
    p.add_argument("--sigreg_target", type=str, default="z", choices=["z", "mu", "mu_feat"],
                   help="z/mu: image-axis SIGReg (B samples in D-space). "
                        "mu_feat: feature-axis SIGReg — transposes mu to [D, B], "
                        "treating D=6144 features as samples and pushing each "
                        "feature's distribution across images toward N(0,1).")
    p.add_argument("--ep_num_points", type=int, default=33)
    p.add_argument("--ep_t_min", type=float, default=-5.0)
    p.add_argument("--ep_t_max", type=float, default=5.0)
    p.add_argument("--ep_no_weight", action="store_true",
                   help="Disable exp(-0.5*t^2) weighting in EP-SIGReg")
    p.add_argument("--ep_slice_chunk_size", type=int, default=256)

    # checkpointing: keep only the best validation checkpoint to avoid quota blowups.
    p.add_argument("--ckpt_dir",   type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=10)
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
        linear_decoder=args.linear_decoder,
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
                if args.kl_type == "sample_l1":
                    kl_base = kl_bottleneck_loss(out["mu"], out["logvar"])
                    l1_mu   = l1_mu_loss(out["mu"])
                    kl      = kl_base + args.alpha_l1 * l1_mu
                else:
                    _kl_fn = {"feature": kl_feature_level, "feature_sq": kl_feature_level_sq,
                              "feature_log": kl_feature_level_log}.get(args.kl_type, kl_bottleneck_loss)
                    kl    = _kl_fn(out["mu"], out["logvar"])
                    kl_base = kl
                    l1_mu   = out["mu"].new_zeros(())
                if args.sigreg_target == "mu_feat":
                    sig_target = out["mu"].T   # [D, B]: feature-axis
                else:
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
            _mu_batch_mean_abs = out["mu"].detach().mean(dim=0).abs().mean()
            for k, v in [
                ("rec_loss", rec), ("kl_loss", kl), ("kl_base_loss", kl_base),
                ("l1_mu_loss", l1_mu),
                ("sigreg_loss", sig),
                ("ep_sigreg_loss", sig if args.sigreg_type == "epps_pulley" else sig.detach() * 0.0),
                ("total_loss", rec + beta_kl_now * kl + args.lambda_sigreg * sig),
                ("mu_sq_mean", out["mu"].pow(2).mean().detach()),
                ("exp_logvar_mean", out["logvar"].exp().mean().detach()),
                ("mu_batch_mean_abs", _mu_batch_mean_abs),
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
        # per-image KL selectivity accumulators
        sel_feat_counts = {thr: None for thr in dead_thresholds}  # [D] sum over images
        sel_img_lists   = {thr: [] for thr in dead_thresholds}    # list of [B] tensors
        sel_total_imgs  = 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(x)
                    rec = reconstruction_loss(out["x_hat"], x)
                    if args.kl_type == "sample_l1":
                        kl_base_v = kl_bottleneck_loss(out["mu"], out["logvar"])
                        l1_mu_v   = l1_mu_loss(out["mu"])
                        kl        = kl_base_v + args.alpha_l1 * l1_mu_v
                    else:
                        _kl_fn = {"feature": kl_feature_level, "feature_sq": kl_feature_level_sq,
                                  "feature_log": kl_feature_level_log}.get(args.kl_type, kl_bottleneck_loss)
                        kl        = _kl_fn(out["mu"], out["logvar"])
                        kl_base_v = kl
                        l1_mu_v   = out["mu"].new_zeros(())
                    if args.sigreg_target == "mu_feat":
                        sig = sigreg_module(out["mu"].T)
                    else:
                        sig = sigreg_module(out[args.sigreg_target])

                kl_sample = kl_bottleneck_loss(out["mu"], out["logvar"])
                batch_health = latent_batch_metrics(out["mu"], out["logvar"], out["z"])
                for k, v in [
                    ("rec_loss", rec), ("kl_loss", kl), ("kl_base_loss", kl_base_v),
                    ("l1_mu_loss", l1_mu_v), ("kl_sample_loss", kl_sample),
                    ("sigreg_loss", sig),
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

                # per-image KL activity: kl_ij = 0.5*(mu_ij^2 + exp(logvar_ij) - logvar_ij - 1)
                mu_f = out["mu"].float()
                lv_f = out["logvar"].float()
                kl_per_img = 0.5 * (mu_f.pow(2) + lv_f.exp() - lv_f - 1)  # [B, D]
                for thr in dead_thresholds:
                    active_mask = kl_per_img.gt(thr)                        # [B, D] bool
                    feat_sum = active_mask.sum(0).float().cpu()              # [D]
                    img_sum  = active_mask.sum(1).float().cpu()              # [B]
                    sel_feat_counts[thr] = feat_sum if sel_feat_counts[thr] is None \
                                           else sel_feat_counts[thr] + feat_sum
                    sel_img_lists[thr].append(img_sum)
                sel_total_imgs += mu_f.shape[0]

        val_metrics = {k: v / n_val for k, v in val_acc.items()}

        # selective activity metrics (per-image KL threshold)
        for thr in dead_thresholds:
            if sel_feat_counts[thr] is not None and sel_total_imgs > 0:
                feat_freq   = sel_feat_counts[thr] / sel_total_imgs
                img_counts  = torch.cat(sel_img_lists[thr])
                sel_metrics = compute_selective_activity(
                    feat_freq, img_counts, thr, latent_dim=args.latent_dim
                )
                val_metrics.update(sel_metrics)

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
            # selective activity (primary threshold)
            **{k: v for k, v in val_metrics.items() if k.startswith("sel/")},
            "epoch_time_sec": elapsed,
        }
        wandb.log(log, step=epoch)

        # pick primary threshold tag for print
        _ptag = f"{args.dead_activation_threshold:g}".replace(".", "_").replace("-", "m")
        print(
            f"[{epoch:04d}/{args.epochs}] "
            f"lr={lr_now:.2e}  β={beta_kl_now:.2e}  "
            f"rec={val_metrics['rec_loss']:.4f}  "
            f"kl={val_metrics['kl_loss']:.1f}  "
            f"kl_s={val_metrics.get('kl_sample_loss', float('nan')):.2f}  "
            f"sig={val_metrics['sigreg_loss']:.4f}  "
            f"total={val_metrics['total_loss']:.4f}  "
            f"dead={val_metrics.get(f'sel/feat_never_active_{_ptag}', float('nan')):.3f}  "
            f"always={val_metrics.get(f'sel/feat_always_active_{_ptag}', float('nan')):.3f}  "
            f"img_frac={val_metrics.get(f'sel/img_active_frac_{_ptag}', float('nan')):.3f}  "
            f"freq_p50={val_metrics.get(f'sel/feat_freq_p50_{_ptag}', float('nan')):.3f}  "
            f"({elapsed:.1f}s)"
        )

        # ---- Checkpoint ----
        is_best = val_metrics["total_loss"] < best_val_total
        if is_best:
            best_val_total = val_metrics["total_loss"]
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "args": vars(args)},
                ckpt_dir / "best.pt",
            )

    # always save final epoch
    torch.save(
        {"epoch": epoch, "model": model.state_dict(),
         "optimizer": optimizer.state_dict(), "args": vars(args)},
        ckpt_dir / "last.pt",
    )

    # ------------------------------------------------------------------ #
    # End-of-training diagnostics: collect full val set, run full suite  #
    # ------------------------------------------------------------------ #
    print("\n[eot] collecting full val-set diagnostics...", flush=True)
    _all_z, _all_mu, _all_lgv = [], [], []
    model.eval()
    with torch.no_grad():
        for (x,) in val_loader:
            x = x.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(x)
            _all_z.append(out["z"].float().cpu())
            _all_mu.append(out["mu"].float().cpu())
            _all_lgv.append(out["logvar"].float().cpu())

    _z_all  = torch.cat(_all_z,  dim=0).to(device)
    _mu_all = torch.cat(_all_mu, dim=0).to(device)
    _lgv_all= torch.cat(_all_lgv,dim=0).to(device)

    _fd   = compute_z_frechet_distance(_z_all)
    _kl   = compute_kl_active_dims(_mu_all, _lgv_all, thresholds=(0.1, 0.5, 1.0))
    _mom  = compute_marginal_moments(_z_all)

    # selective activity on full val set
    _eot_sel: dict = {}
    _eot_sel_lines: list[str] = []
    for _thr in [0.1, 0.5, 1.0]:
        _kl_per_img = 0.5 * (_mu_all.pow(2) + _lgv_all.exp() - _lgv_all - 1)  # [N, D]
        _active     = _kl_per_img.gt(_thr)
        _feat_freq  = _active.float().mean(0).cpu()   # [D]
        _img_counts = _active.float().sum(1).cpu()    # [N]
        _sm = compute_selective_activity(_feat_freq, _img_counts, _thr, args.latent_dim)
        _eot_sel.update(_sm)
        _tag = f"{_thr:g}".replace(".", "_").replace("-", "m")
        _eot_sel_lines += [
            f"  threshold={_thr}",
            f"    feat_never_active  = {_sm[f'sel/feat_never_active_{_tag}']:.4f}  (dead; target→0)",
            f"    feat_always_active = {_sm[f'sel/feat_always_active_{_tag}']:.4f}  (saturated; target→0)",
            f"    feat_freq_p10/p50/p90 = "
            f"{_sm[f'sel/feat_freq_p10_{_tag}']:.3f} / "
            f"{_sm[f'sel/feat_freq_p50_{_tag}']:.3f} / "
            f"{_sm[f'sel/feat_freq_p90_{_tag}']:.3f}",
            f"    img_active_frac    = {_sm[f'sel/img_active_frac_{_tag}']:.4f}  (sparse→low)",
            f"    img_active_mean    = {_sm[f'sel/img_active_mean_{_tag}']:.1f}  features/image",
        ]

    # decoder weight norm per latent dim (linear decoder only)
    _dec_lines = []
    _dec_metrics = {}
    if hasattr(model, "decoder") and hasattr(model.decoder, "weight"):
        _col_norms = model.decoder.weight.float().norm(dim=0).cpu()  # [D]
        _thr_vals = [0.01, 0.05, 0.1, 0.5]
        _max_norm = _col_norms.max().item()
        _dec_metrics = {
            "dec_col_norm_mean":   _col_norms.mean().item(),
            "dec_col_norm_max":    _max_norm,
            "dec_col_norm_std":    _col_norms.std().item(),
        }
        for _t in _thr_vals:
            _abs_thr = _t * _max_norm
            _n = (_col_norms > _abs_thr).sum().item()
            _dec_metrics[f"dec_active_norm_{str(_t).replace('.','p')}"] = _n
        _thr_lines = []
        for _t in _thr_vals:
            _key = "dec_active_norm_" + str(_t).replace(".", "p")
            _thr_lines.append(
                f"  dims with norm > {_t:.0%}·max: {_dec_metrics[_key]} / {args.latent_dim}"
            )
        _dec_lines = [
            "",
            "## Decoder Column Norms  (linear decoder; proxy for latent dim usage)",
            f"  mean = {_dec_metrics['dec_col_norm_mean']:.4f}",
            f"  std  = {_dec_metrics['dec_col_norm_std']:.4f}",
            f"  max  = {_dec_metrics['dec_col_norm_max']:.4f}",
        ] + _thr_lines

    eot = {**_fd, **_kl, **_mom, **_eot_sel, **_dec_metrics}
    wandb.summary.update({f"eot/{k}": v for k, v in eot.items()})

    _report_lines = [
        "# Stage-1 End-of-Training Diagnostics",
        (
            f"beta_kl={args.beta_kl}  lambda_sigreg={args.lambda_sigreg}"
            f"  kl_warmup={args.beta_kl_warmup_epochs}  epochs={args.epochs}"
        ),
        "",
        "## z-space Frechet Distance  (lower → closer to N(0,I))",
        f"  z_fd_diag = {_fd['z_fd_diag']:.4f}   [||µ||² + Σ(σᵢ-1)²]",
        f"  z_fd_proj = {_fd['z_fd_proj']:.6f}  [sliced, 512 projections]",
        "",
        "## KL Active Dimensions",
        f"  kl_total   = {_kl['kl_total']:.1f}",
        f"  active@0.1 = {int(_kl['active_dims_at_0_1'])}",
        f"  active@0.5 = {int(_kl['active_dims_at_0_5'])}",
        f"  active@1.0 = {int(_kl['active_dims_at_1'])}",
        "",
        "## Selective Activity (per-image KL threshold)",
        *_eot_sel_lines,
        "",
        "## Marginal Moments",
        f"  mean_abs_mean  = {_mom['marginal_mean_abs_mean']:.4f}   (target 0)",
        f"  std_mean       = {_mom['marginal_std_mean']:.4f}   (target 1)",
        f"  std_std        = {_mom['marginal_std_std']:.4f}   (target 0)",
        f"  skew_abs_mean  = {_mom['marginal_skew_abs_mean']:.4f}   (target 0)",
        f"  kurt_abs_mean  = {_mom['marginal_excess_kurt_abs_mean']:.4f}   (target 0)",
        *_dec_lines,
    ]
    _report_text = "\n".join(_report_lines) + "\n"
    _report_path = ckpt_dir / "diagnostics_report.md"
    _report_path.write_text(_report_text)
    print(_report_text, flush=True)
    print(f"[eot] report → {_report_path}", flush=True)

    samples = sample_from_prior(model, 8, args.latent_dim, device)
    print(f"Prior samples: {samples.shape}")
    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
