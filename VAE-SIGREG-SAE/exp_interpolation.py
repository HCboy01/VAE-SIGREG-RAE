"""
Latent space interpolation between image pairs.
z = (1-t)*mu_A + t*mu_B, decode at each step.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from src.vae_sigreg import OvercompleteVariationalAE


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ckpt['args']
    model = OvercompleteVariationalAE(
        a['input_dim'], a['latent_dim'], a.get('hidden_dim'), a['num_layers']
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt['epoch']


@torch.no_grad()
def encode_all(model, embeddings, device, batch_size=1024):
    mus = []
    for i in range(0, len(embeddings), batch_size):
        mu, _ = model.encode(embeddings[i:i+batch_size].to(device))
        mus.append(mu.cpu())
    return torch.cat(mus)


def load_image(path, size=160):
    return Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, epoch = load_model('checkpoints/epoch_0200.pt', device)
    print(f'Model epoch={epoch}')

    shape = np.load('/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features_shape.npy')
    emb = torch.from_numpy(
        np.fromfile('/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features.bin',
                    dtype=np.float32).reshape(shape)
    )
    paths = [l.strip() for l in open('/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_paths.txt')]

    print('Encoding...')
    mu = encode_all(model, emb, device)
    kl = 0.5 * mu.pow(2).sum(dim=1)   # approx total KL (var≈1이므로)
    kl_per_image = kl

    # 다양한 쌍 선택
    idx_max  = kl_per_image.argmax().item()
    idx_min  = kl_per_image.argmin().item()
    idx_med  = (kl_per_image - kl_per_image.median()).abs().argmin().item()
    torch.manual_seed(0)
    rand_idx = torch.randperm(len(emb))[:4].tolist()

    pairs = [
        ('max_KL ↔ min_KL',    idx_max,      idx_min),
        ('max_KL ↔ median',     idx_max,      idx_med),
        ('random pair 1',       rand_idx[0],  rand_idx[1]),
        ('random pair 2',       rand_idx[2],  rand_idx[3]),
    ]

    steps = 10   # 0 to 1 inclusive → 11 frames
    t_vals = torch.linspace(0, 1, steps + 1)

    Path('visualizations').mkdir(exist_ok=True)

    fig, axes = plt.subplots(len(pairs), steps + 1, figsize=((steps + 1) * 2, len(pairs) * 2.2))
    fig.suptitle(f'Latent interpolation  (mu space, epoch={epoch})', fontsize=13, fontweight='bold')

    for row, (label, idx_a, idx_b) in enumerate(pairs):
        mu_a = mu[idx_a].to(device)
        mu_b = mu[idx_b].to(device)

        img_a = load_image(paths[idx_a])
        img_b = load_image(paths[idx_b])

        with torch.no_grad():
            for col, t in enumerate(t_vals):
                z = (1 - t) * mu_a + t * mu_b          # linear interpolation
                x_hat = model.decode(z.unsqueeze(0))    # [1, input_dim]

                # DINO embedding을 직접 이미지로 바꿀 수 없으므로
                # → 원본 이미지 기반 시각화 (t=0, t=1) + 중간은 mu 거리 표시
                ax = axes[row, col]
                if col == 0:
                    ax.imshow(img_a)
                    ax.set_title(f't=0', fontsize=8)
                elif col == steps:
                    ax.imshow(img_b)
                    ax.set_title(f't=1', fontsize=8)
                else:
                    # 중간 지점: nearest neighbor in mu space
                    z_cpu = z.cpu()
                    dists = (mu - z_cpu.unsqueeze(0)).pow(2).sum(dim=1)
                    # exclude endpoints
                    dists[idx_a] = 1e9
                    dists[idx_b] = 1e9
                    nn_idx = dists.argmin().item()
                    nn_img = load_image(paths[nn_idx])
                    ax.imshow(nn_img)
                    ax.set_title(f't={t:.1f}', fontsize=8)
                ax.axis('off')

        axes[row, 0].set_ylabel(label, fontsize=9, rotation=90, labelpad=5)

    plt.tight_layout()
    out = 'visualizations/interpolation.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
