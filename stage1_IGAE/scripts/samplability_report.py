"""
Is the latent samplable from N(0, I)?

Encodes N images, draws z = mu + sigma*eps, and measures how far the
aggregated posterior q(z) is from the prior N(0, I) using the existing
diagnostics:
  - z_fd_diag / z_fd_proj  (Frechet distance; proj captures off-diagonal cov)
  - off_diag_cov_mean/max  (cross-dim correlations -> joint Gaussianity)
  - skewness / excess kurtosis (marginal shape beyond 2nd moment)

Reports each metric on ALL dims and on the ACTIVE subset (KL>0.1), since the
prior mismatch lives in the active dims (inactive dims are ~N(0,1) by default).

Usage:
    python scripts/samplability_report.py RUN [RUN ...] [--n 3000]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.vae_sigreg import OvercompleteVariationalAE
from src.vae_sigreg.diagnostics import (
    compute_latent_diagnostics, compute_z_frechet_distance)

CKPT_ROOT = Path("/scratch/x3411a10/IGAE/stage1_IGAE/checkpoints")
DATA_PATH = Path("/scratch/x3411a10/datasets/ffhq256/dino_features_train.pt")
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--n", type=int, default=3000)
    return p.parse_args()


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    is_linear = "decoder.weight" in ckpt["model"]
    model = OvercompleteVariationalAE(
        input_dim=a["input_dim"], latent_dim=a["latent_dim"],
        hidden_dim=a.get("hidden_dim"), num_layers=a["num_layers"],
        linear_decoder=is_linear,
    )
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, a


@torch.no_grad()
def encode_all(model, x, device, bs=512):
    mus, lvs = [], []
    for i in range(0, len(x), bs):
        mu, lv = model.encode(x[i:i+bs].to(device))
        mus.append(mu.cpu()); lvs.append(lv.cpu())
    return torch.cat(mus), torch.cat(lvs)


def report_block(tag, z):
    fd = compute_z_frechet_distance(z)
    d  = compute_latent_diagnostics(z)
    print(f"  [{tag}]  D={z.shape[1]}")
    print(f"    z_fd_diag           = {fd['z_fd_diag']:.3f}   "
          f"(||mean||² + Σ(std-1)²; 0 = perfect)")
    print(f"    z_fd_proj           = {fd['z_fd_proj']:.4f}   "
          f"(sliced; catches off-diag cov)")
    print(f"    per-dim |mean|      = {d['mean_abs_mean']:.4f}")
    print(f"    per-dim |var-1|     = {d['var_err_from_1']:.4f}")
    print(f"    off_diag_cov mean   = {d['off_diag_cov_mean']:.4f}   "
          f"max = {d['off_diag_cov_max']:.4f}")
    print(f"    |skew| mean         = {d['skewness_abs_mean']:.4f}")
    print(f"    |excess kurt| mean  = {d['excess_kurtosis_abs_mean']:.4f}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    raw  = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    data = raw.float() if isinstance(raw, torch.Tensor) else next(iter(raw.values())).float()
    torch.manual_seed(SEED)
    x = data[torch.randperm(len(data))[:args.n]]

    for run in args.runs:
        model, a = load_model(CKPT_ROOT / run / "best.pt")
        model.to(device)
        mu, logvar = encode_all(model, x, device)
        model.cpu()
        sigma = (0.5 * logvar).exp()
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(0)

        torch.manual_seed(SEED)
        z = mu + sigma * torch.randn_like(mu)

        active = kl > 0.1
        n_act = int(active.sum())

        print("=" * 64)
        print(f"{run}   β={a.get('beta_kl')}  λ={a.get('lambda_sigreg')}  "
              f"N={args.n}   active@0.1 = {n_act}/{kl.numel()}")
        report_block("ALL dims", z)
        if 0 < n_act < kl.numel():
            report_block("ACTIVE dims (KL>0.1)", z[:, active])
        print()


if __name__ == "__main__":
    main()
