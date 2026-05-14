"""
Active Unit analysis.
An "active unit" is a neuron j where mean_images(KL_j) > threshold.
Measures how many latent dimensions actually encode information.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    mus, lvs = [], []
    for i in range(0, len(embeddings), batch_size):
        mu, lv = model.encode(embeddings[i:i+batch_size].to(device))
        mus.append(mu.cpu()); lvs.append(lv.cpu())
    return torch.cat(mus), torch.cat(lvs)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, epoch = load_model('checkpoints/epoch_0200.pt', device)
    print(f'Model epoch={epoch}')

    shape = np.load('/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features_shape.npy')
    emb = torch.from_numpy(
        np.fromfile('/workspace/hyeongchan/datasets/ffhq256/dino_cls_features/train_features.bin',
                    dtype=np.float32).reshape(shape)
    )

    print('Encoding...')
    mu, lv = encode_all(model, emb, device)
    kl = 0.5 * (mu.pow(2) + lv.exp() - lv - 1)   # [N, D]

    # Per-neuron average KL across all images
    kl_per_neuron = kl.mean(dim=0)    # [D]
    # Per-image total KL
    kl_per_image  = kl.sum(dim=1)     # [N]
    # Per-image active unit count (KL_j > threshold)
    D = mu.shape[1]

    thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]

    print(f'\n=== Active Unit Analysis (D={D}, N={len(mu)}) ===')
    print(f'{"Threshold":>12} | {"Active neurons":>15} | {"% of D":>8} | {"Avg active/image":>17}')
    print('-' * 60)
    for thr in thresholds:
        # neuron j is "active" if mean_i(KL_ij) > thr
        active_neurons = (kl_per_neuron > thr).sum().item()
        # per image: how many neurons have KL_j > thr for that image
        active_per_img = (kl > thr).float().sum(dim=1).mean().item()
        print(f'{thr:>12.3f} | {active_neurons:>15,} | {active_neurons/D*100:>7.1f}% | {active_per_img:>17.1f}')

    Path('visualizations').mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Active Unit Analysis (epoch={epoch}, beta_kl=1e-2, D={D})', fontsize=13, fontweight='bold')

    # 1) Per-neuron mean KL distribution
    axes[0].hist(kl_per_neuron.numpy(), bins=200, color='steelblue', alpha=0.8)
    for thr, c in zip([0.001, 0.01, 0.05], ['orange', 'red', 'darkred']):
        axes[0].axvline(thr, color=c, lw=1.5, ls='--', label=f'thr={thr}')
    axes[0].set_title('Per-neuron mean KL\n(averaged over all images)', fontsize=11)
    axes[0].set_xlabel('mean KL per neuron')
    axes[0].set_ylabel('count')
    axes[0].legend(fontsize=8)
    axes[0].set_yscale('log')

    # 2) Cumulative active unit curve
    sorted_kl = kl_per_neuron.sort(descending=True).values.numpy()
    axes[1].plot(np.arange(1, D+1), sorted_kl, color='steelblue', lw=1.5)
    for thr, c in zip([0.001, 0.01, 0.05], ['orange', 'red', 'darkred']):
        n_active = (kl_per_neuron > thr).sum().item()
        axes[1].axhline(thr, color=c, lw=1, ls='--', label=f'thr={thr} → {n_active} neurons')
        axes[1].axvline(n_active, color=c, lw=1, ls=':')
    axes[1].set_title('Sorted per-neuron mean KL\n(cumulative)', fontsize=11)
    axes[1].set_xlabel('neuron rank (by KL)')
    axes[1].set_ylabel('mean KL')
    axes[1].legend(fontsize=8)
    axes[1].set_yscale('log')

    # 3) Per-image active unit count (thr=0.01)
    thr = 0.01
    active_per_img = (kl > thr).float().sum(dim=1).numpy()
    axes[2].hist(active_per_img, bins=80, color='coral', alpha=0.8)
    axes[2].axvline(active_per_img.mean(), color='black', lw=1.5, ls='--',
                    label=f'mean={active_per_img.mean():.1f}')
    axes[2].set_title(f'Active neurons per image\n(KL > {thr})', fontsize=11)
    axes[2].set_xlabel('# active neurons per image')
    axes[2].set_ylabel('count')
    axes[2].legend()

    plt.tight_layout()
    out = 'visualizations/active_units.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nSaved → {out}')


if __name__ == '__main__':
    main()
