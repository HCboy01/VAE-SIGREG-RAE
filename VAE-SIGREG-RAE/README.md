# VAE-SIGREG-RAE

Stage-2 RAE DiT fine-tuning with **DINO CLS -> frozen VAE-SIGREG encoder -> mu latent** conditioning.

The default conditioner uses:

```text
/scratch/x3411a10/VAE-SIGREG/VAE-SIGREG-SAE/checkpoints/200epoch.pt
```

and produces a 3072-dimensional `mu` vector that is injected into the DiT time-conditioning stream through a zero-initialized adapter.

## Key Files

- `src/vae_rae/conditioning.py`
  - `VaeSigregConditioner`
  - Loads DINOv2 and the frozen VAE-SIGREG checkpoint, then returns `mu`.
- `src/vae_rae/models/ddt_vae.py`
  - `DiTwDDTHeadVAECond`
  - Adds `cond_proj` and `cond_gate` to the RAE Stage-2 DiT.
- `src/train_vae_cond.py`
  - Single-GPU fine-tuning entrypoint.
- `src/cache_vae_rae_latents.py`
  - Optional offline cache builder for RAE `z` and VAE-SIGREG `mu` condition vectors.
- `configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_VAESIGREG.yaml`
  - Default training config.

## Train

```bash
cd /scratch/x3411a10/VAE-SIGREG/VAE-SIGREG-RAE
export EXPERIMENT_NAME=vae_sigreg_mu_cond
python src/train_vae_cond.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_VAESIGREG.yaml \
  --data-path /scratch/x3411a10/datasets/ffhq256/imagefolder/train \
  --results-dir ckpts \
  --image-size 256 \
  --precision bf16 \
  --workers 8
```

## Optional Cache

```bash
cd /scratch/x3411a10/VAE-SIGREG/VAE-SIGREG-RAE
python src/cache_vae_rae_latents.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_VAESIGREG.yaml \
  --data-path /scratch/x3411a10/datasets/ffhq256/imagefolder/train \
  --output-dir cache/ffhq256_vae_sigreg_mu_aug2 \
  --image-size 256 \
  --batch-size 32 \
  --precision bf16 \
  --num-augs 2

python src/train_vae_cond.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_VAESIGREG.yaml \
  --data-path /scratch/x3411a10/datasets/ffhq256/imagefolder/train \
  --cached-latents-dir cache/ffhq256_vae_sigreg_mu_aug2 \
  --results-dir ckpts \
  --image-size 256 \
  --precision bf16 \
  --workers 8
```
