# IGAE Stage 2: Conditioned RAE

Stage 2 RAE DiT fine-tuning with **DINOv2 CLS → frozen IGAE encoder → mu latent** conditioning.

The default conditioner loads:

```text
stage1_IGAE/best_ckpt/beta_1e-3_lambda_10_best.pt
```

and produces a 3072-dimensional `mu` vector injected into the DiT time-conditioning stream via a zero-initialized adapter.

## Key Files

- `src/vae_rae/conditioning.py`
  - `VaeSigregConditioner`: loads DINOv2 and the frozen IGAE checkpoint, returns `mu`.
- `src/vae_rae/models/ddt_vae.py`
  - `DiTwDDTHeadVAECond`: adds `cond_proj` and `cond_gate` to the RAE Stage-2 DiT.
- `src/train_vae_cond.py`
  - Single-GPU fine-tuning entrypoint.
- `src/cache_vae_rae_latents.py`
  - Optional offline cache builder for RAE `z` and IGAE `mu` condition vectors.
- `configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_VAESIGREG.yaml`
  - Default training config.

## Train

```bash
cd stage2_conditioned_RAE
export EXPERIMENT_NAME=igae_mu_cond
python src/train_vae_cond.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_VAESIGREG.yaml \
  --data-path /path/to/ffhq256/imagefolder/train \
  --results-dir ckpts \
  --image-size 256 \
  --precision bf16 \
  --workers 8
```

## Optional Cache

```bash
cd stage2_conditioned_RAE
python src/cache_vae_rae_latents.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_VAESIGREG.yaml \
  --data-path /path/to/ffhq256/imagefolder/train \
  --output-dir cache/ffhq256_igae_mu_aug2 \
  --image-size 256 \
  --batch-size 32 \
  --precision bf16 \
  --num-augs 2

python src/train_vae_cond.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_VAESIGREG.yaml \
  --data-path /path/to/ffhq256/imagefolder/train \
  --cached-latents-dir cache/ffhq256_igae_mu_aug2 \
  --results-dir ckpts \
  --image-size 256 \
  --precision bf16 \
  --workers 8
```
