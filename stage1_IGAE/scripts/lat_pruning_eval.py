"""
Post-hoc latent pruning for a trained VAE, by replacing low-importance dims
with their PRIOR.

Ranks latent dims by an importance criterion, then for the least-important k
dims overrides the reparameterized z_d with a standard-normal sample
(z_d ~ N(0,1)) -- the prior. Kept dims use the posterior mean mu. No retraining.

Why prior (N(0,1)) and not 0:
  The generative process samples z ~ N(0,I), so "dropping" a dim means a
  generator feeds it N(0,1), not the constant 0. A dead dim has posterior ~=
  prior (mu~0, sigma~1), so the decoder only ever saw z_d ~ N(0,1) there during
  training -- feeding 0 is off-distribution. Replacing a dim with its prior
  therefore tests the real question: "can Stage-2 just sample this dim from
  N(0,1)?" (the samplability question). Truly prior-like dims cost ~nothing.

Two curves are reported at every prune level (kept dims = mu both times):
  rec(prior) : pruned dims ~ N(0,1)   -- signal loss + injected prior noise
  rec(zero)  : pruned dims = 0         -- signal loss only (denoising/MAP view)
gap = rec(prior) - rec(zero) is the noise the pruned dims' decoder columns
inject when sampled from the prior. For dims that are already prior-like with
small decoder columns, both curves stay flat -> safe to prune.

Criteria (--criterion):
  kl       : per-dim mean KL  0.5*E[mu^2 + sigma^2 - logvar - 1]  (default;
             the principled match for prior-fill -- KL_d is exactly the
             posterior-vs-prior divergence, i.e. the cost of replacing dim d
             with its prior. KL~0 => posterior already ~= prior => free to prune)
  l1       : per-dim mean absolute posterior mean  E[|mu_d|]
  dec_norm : decoder column L2 norm  ||W[:,d]||  (reconstruction sensitivity;
             linear decoder only)
  mu_var   : Var_n[mu[:,d]]  (how much the dim's mean moves across images)

Usage:
    python scripts/lat_pruning_eval.py [RUN] [--criterion dec_norm] [--n 5000] [--steps 25]
Output:
    visualizations/pruning/<run>_prune_<criterion>.png
    visualizations/pruning/<run>_prune_<criterion>.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vae_sigreg.model import OvercompleteVariationalAE

CKPT_ROOT = Path("/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints")
EMB_PATH  = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
OUT_DIR   = Path("/scratch/x3411a10/IGAE/stage1_IGAE/visualizations/pruning")

SEED = 42
BATCH_SIZE = 512
AMP_DTYPE = torch.bfloat16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", default="s1_grid_b1e-3_l100_lindec")
    p.add_argument("--criterion", choices=["kl", "l1", "dec_norm", "mu_var"], default="kl")
    p.add_argument("--n", type=int, default=5000, help="num images to evaluate on")
    p.add_argument("--steps", type=int, default=25, help="num prune levels to sweep")
    p.add_argument("--tail_keep_max", type=int, default=512,
                   help="densely evaluate the right-tail region with dims kept <= this value")
    p.add_argument("--tail_keep_step", type=int, default=8,
                   help="step size for dense right-tail evaluation in dims kept")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--fig_width", type=float, default=19.0, help="output figure width in inches")
    p.add_argument("--fig_height", type=float, default=5.5, help="output figure height in inches")
    return p.parse_args()


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    state = ckpt["model"]
    # The state dict is the source of truth. Some older runs may have args
    # recorded under defaults that later changed.
    linear_decoder = "decoder.weight" in state
    use_reparam = not bool(a.get("no_reparameterization", False) or a.get("deterministic_ae", False))
    model = OvercompleteVariationalAE(
        input_dim=int(a.get("input_dim", 768)),
        latent_dim=int(a.get("latent_dim", 6144)),
        hidden_dim=a.get("hidden_dim", None),
        num_layers=int(a.get("num_layers", 4)),
        linear_decoder=linear_decoder,
        use_reparameterization=use_reparam,
    ).to(device)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    return model, a, linear_decoder


@torch.no_grad()
def encode_all(model, emb, device):
    mus, lvs = [], []
    for i in range(0, len(emb), BATCH_SIZE):
        x = emb[i:i+BATCH_SIZE].to(device)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
            mu, lv = model.encode(x)
        mus.append(mu.float().cpu()); lvs.append(lv.float().cpu())
    return torch.cat(mus), torch.cat(lvs)


@torch.no_grad()
def decode_mse(model, mu, inp, keep_mask, device, fill, generator=None):
    """
    Mean MSE when dims outside keep_mask are replaced. Kept dims use mu.
      fill="prior": pruned dims drawn from N(0,1)  (the prior)
      fill="zero" : pruned dims set to 0           (denoising / MAP view)
    """
    keep = keep_mask.to(device)
    drop = 1.0 - keep
    se_sum, n = 0.0, 0
    for i in range(0, len(mu), BATCH_SIZE):
        m = mu[i:i+BATCH_SIZE].to(device)
        z = m * keep  # keep posterior mean on retained dims
        if fill == "prior":
            eps = torch.randn(m.shape, device=device, generator=generator)
            z = z + eps * drop  # prior sample on pruned dims
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
            x_hat = model.decode(z).float()
        se_sum += F.mse_loss(x_hat, inp[i:i+BATCH_SIZE].to(device), reduction="sum").item()
        n += x_hat.numel()
    return se_sum / n


def importance_scores(model, mu, logvar, criterion, linear_decoder):
    if criterion == "dec_norm":
        if not (linear_decoder and hasattr(model.decoder, "weight")):
            raise SystemExit("dec_norm criterion needs a linear decoder")
        return model.decoder.weight.float().norm(dim=0).cpu()  # [D]
    if criterion == "kl":
        return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(0)  # [D]
    if criterion == "l1":
        return mu.abs().mean(0)  # [D]
    if criterion == "mu_var":
        return mu.var(0, unbiased=False)  # [D]
    raise ValueError(criterion)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  run: {args.run}  criterion: {args.criterion}")

    obj = torch.load(EMB_PATH, map_location="cpu", weights_only=False)
    emb = (obj if isinstance(obj, torch.Tensor) else next(iter(obj.values()))).float()
    torch.manual_seed(args.seed)
    sub = torch.randperm(len(emb))[:min(args.n, len(emb))]
    emb = emb[sub].contiguous()

    model, a, linear_decoder = load_model(CKPT_ROOT / args.run / "best.pt", device)
    D = int(a.get("latent_dim", 6144))
    beta, lam = a.get("beta_kl", "?"), a.get("lambda_sigreg", "?")
    print(f"  beta_kl={beta}  lambda_sigreg={lam}  D={D}  N={len(emb)}")

    mu, logvar = encode_all(model, emb, device)
    scores = importance_scores(model, mu, logvar, args.criterion, linear_decoder)

    # rank ascending: lowest score = pruned first
    order = scores.argsort()  # least important -> most important

    # prune levels (#dims removed): geometric near 0 (where dead dims live),
    # linear coverage over the full range, and dense coverage in the low-kept
    # tail where collapse can happen quickly.
    geo = np.geomspace(1, D - 1, args.steps).round().astype(int)
    lin = np.linspace(0, D - 1, args.steps).round().astype(int)
    tail_kept = np.arange(1, min(args.tail_keep_max, D - 1) + 1,
                          max(1, args.tail_keep_step))
    tail = D - tail_kept
    n_pruned = np.unique(np.concatenate([[0], geo, lin, tail]))

    gen = torch.Generator(device=device).manual_seed(args.seed)
    rows = []
    print(f"\n{'pruned':>7} {'kept':>6} {'keep%':>6} "
          f"{'rec(prior)':>11} {'rec(zero)':>10} {'gap':>9}")
    for k in n_pruned:
        keep_mask = torch.ones(D)
        if k > 0:
            keep_mask[order[:k]] = 0.0
        rec_pr = decode_mse(model, mu, emb, keep_mask, device, fill="prior", generator=gen)
        rec_z0 = decode_mse(model, mu, emb, keep_mask, device, fill="zero")
        kept = D - int(k)
        rows.append((int(k), kept, kept / D, rec_pr, rec_z0, rec_pr - rec_z0))
        print(f"{int(k):>7} {kept:>6} {kept/D:>6.1%} "
              f"{rec_pr:>11.5f} {rec_z0:>10.5f} {rec_pr-rec_z0:>9.5f}")

    rows = np.array(rows, dtype=float)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / f"{args.run}_prune_{args.criterion}"
    np.savetxt(f"{base}.csv", rows, delimiter=",",
               header="n_pruned,n_kept,keep_frac,rec_prior,rec_zero,prior_zero_gap", comments="")

    rec0 = rows[0, 3]  # k=0: prior==zero==decode(mu) baseline

    fig, axes = plt.subplots(1, 2, figsize=(args.fig_width, args.fig_height))
    ax = axes[0]
    ax.plot(rows[:, 1], rows[:, 3], "-o", ms=3, label="rec(prune->prior N(0,1))")
    ax.plot(rows[:, 1], rows[:, 4], "-s", ms=3, label="rec(prune->zero)")
    ax.axhline(rec0, ls="--", c="gray", lw=0.8, label="baseline decode(mu)")
    ax.set_xlabel("dims kept (high importance retained)")
    ax.set_ylabel("reconstruction MSE")
    ax.set_title(f"{args.run}\nbeta={beta} lambda={lam}  rank by {args.criterion}")
    ax.invert_xaxis()  # left = many kept, right = heavily pruned
    ax.set_xticks(np.arange(0, D + 1, 512))
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    tail_rows = rows[rows[:, 1] <= args.tail_keep_max]
    ax.plot(tail_rows[:, 1], tail_rows[:, 3], "-o", ms=3, label="rec(prune->prior N(0,1))")
    ax.plot(tail_rows[:, 1], tail_rows[:, 4], "-s", ms=3, label="rec(prune->zero)")
    ax.axhline(rec0, ls="--", c="gray", lw=0.8, label="baseline decode(mu)")
    ax.set_xlabel("dims kept")
    ax.set_ylabel("reconstruction MSE")
    ax.set_title(f"Zoom: dims kept <= {args.tail_keep_max}")
    ax.invert_xaxis()
    tick_step = 64 if args.tail_keep_max <= 512 else 128
    ax.set_xticks(np.arange(0, args.tail_keep_max + 1, tick_step))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # "free prune": most dims replaceable-by-prior while rec(prior) stays within
    # 1% of the baseline decode(mu) -> those dims are effectively prior already.
    free = rows[rows[:, 3] <= 1.01 * rec0]
    if len(free):
        best = free[free[:, 0].argmax()]
        print(f"\nFree-prune (rec(prior) within 1% of baseline {rec0:.5f}): "
              f"prune {int(best[0])} dims, keep {int(best[1])} ({best[2]:.1%}), "
              f"rec(prior)={best[3]:.5f}, gap={best[5]:.5f}")
    print(f"\nSaved -> {base}.png / .csv")


if __name__ == "__main__":
    main()
