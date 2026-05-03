# VAE-SIGREG

DINO 비전 임베딩 위에서 **SIGReg(Sliced Isotropic Gaussian Regularization)** 정규화를 적용한 초과완전(Overcomplete) VAE를 학습하고, 그 잠재 표현을 조건으로 활용해 조건부 이미지 생성 모델(RAE + DiT)을 파인튜닝하는 2단계 생성 모델 파이프라인입니다.

---

## 프로젝트 구조

```
VAE-SIGREG/
├── VAE-SIGREG-SAE/        # Stage 1: Overcomplete VAE with SIGReg
└── VAE-SIGREG-RAE/        # Stage 2: Conditional RAE (DiT-DH fine-tuning)
```

---

## Stage 1: VAE-SIGREG-SAE

### 개요

DINOv2 CLS 임베딩(768-dim)을 입력으로 받아 3072-dim 잠재 공간으로 매핑하는 초과완전 VAE입니다. KL divergence 외에 **SIGReg** 정규화를 추가해 집합 사후 분포(aggregate posterior) `q(z)`가 등방성 가우시안 `N(0, I)`에 가까워지도록 강제합니다.

### 아키텍처

| 구성 요소 | 세부 내용 |
|---|---|
| 입력 | DINO CLS 임베딩 (768-dim) |
| 인코더 | 4층 MLP: 768 → 3072 → 3072 → 3072 → 3072 (LayerNorm + GELU) |
| 잠재 공간 | `mu`, `logvar` 각 3072-dim (overcomplete: latent > input) |
| 디코더 | 대칭 4층 MLP: 3072 → 3072 → 3072 → 3072 → 768 |
| 정밀도 | BFloat16 혼합 정밀도 |

### 손실 함수

```
L_total = L_recon + β_kl × L_KL + λ_sigreg × L_SIGReg
```

| 손실 항 | 설명 |
|---|---|
| `L_recon` | MSE 재구성 손실 |
| `L_KL` | 샘플별 KL divergence (KL annealing 적용, 50에폭에 걸쳐 선형 증가) |
| `L_SIGReg` | 랜덤 투영으로 집합 사후 분포의 평균·분산·왜도·첨도를 N(0,1)에 맞춤 |

**SIGReg 상세**: 랜덤 방향으로 z를 1D 슬라이스로 투영한 뒤 각 슬라이스의 모멘트를 패널티로 부과합니다.
- 평균² → 0
- (분산 − 1)² → 0
- 0.1 × 왜도² → 0
- 0.1 × (첨도 − 3)² → 0

### 학습 설정

| 하이퍼파라미터 | 값 |
|---|---|
| 데이터셋 | FFHQ256 DINO CLS 임베딩 (63,001 train) |
| 에폭 | 200 |
| 배치 사이즈 | 512 (그래디언트 누적 4× → 유효 2048) |
| 학습률 | 3e-4 (코사인 감쇠 + 5에폭 웜업) |
| β_kl | 1e-2 (50에폭 선형 어닐링) |
| λ_sigreg | 0.05 |
| 투영 수 | 3072 |
| optimizer | AdamW (β1=0.9, β2=0.95, weight_decay=1e-2) |

### 학습 실행

```bash
cd VAE-SIGREG-SAE
bash run_train.sh
# 또는 직접 실행
python train_run.py \
    --data_path <TRAIN_FEATURES.bin> \
    --val_data_path <VAL_FEATURES.bin> \
    --input_dim 768 \
    --latent_dim 3072 \
    --num_layers 4 \
    --epochs 200 \
    --batch_size 512 \
    --accum_steps 4 \
    --lr 3e-4 \
    --beta_kl 1e-2 \
    --beta_kl_warmup_epochs 50 \
    --lambda_sigreg 0.05 \
    --num_projections 3072
```

### 체크포인트

`checkpoints/` 에 학습 완료된 체크포인트가 저장됩니다.

| 파일 | β_kl | SIGReg | 비고 |
|---|---|---|---|
| `1e-2_200epoch.pt` | 1e-2 | ✓ | 기본 설정 |
| `1e-3_200epoch.pt` | 1e-3 | ✓ | KL 약화 |
| `3e-5_200epoch.pt` | 3e-5 | ✓ | KL 최소화 |
| `3e-5_200_noSIG_epoch.pt` | 3e-5 | ✗ | SIGReg 없는 ablation |

```python
ckpt = torch.load("checkpoints/1e-3_200epoch.pt", map_location="cpu", weights_only=False)
# ckpt["model"], ckpt["epoch"], ckpt["args"]
model = OvercompleteVariationalAE(**{k: ckpt["args"][k] for k in ["input_dim", "latent_dim", "hidden_dim", "num_layers"]})
model.load_state_dict(ckpt["model"])
```

### 분석 및 시각화 스크립트

| 스크립트 | 설명 |
|---|---|
| `exp_active_units.py` | 차원별 KL 분석 → 활성 잠재 차원 식별 |
| `exp_interpolation.py` | 잠재 공간 보간 실험 |
| `visualize_neuron.py` | 뉴런별 극단 이미지 그리드 (4×4) |
| `visualize_latent.py` | 단일 잠재 축 방향 극단 이미지 |
| `visualize_dist.py` | 뉴런별 mu 분포 히스토그램 |
| `visualize_covariance.py` | 공분산 히트맵 + 인수분해 메트릭 |

### 진단 메트릭 (`diagnostics.py`)

- 차원별 평균 절댓값
- 분산 오차 (목표: 1)
- 비대각 공분산 크기
- 왜도 및 초과 첨도

---

## Stage 2: VAE-SIGREG-RAE

### 개요

Stage 1에서 학습한 VAE의 **잠재 벡터(3072-dim)** 를 조건 신호로 사용해 DiT-DH(Diffusion Transformer with DDT Head) 기반 RAE를 파인튜닝합니다. 이미지 → DINOv2 → 고정된 VAE 인코더 → z = mu + σε → DiT 조건 주입의 흐름으로 동작합니다.

### 조건화 메커니즘

```
Image (256×256)
    ↓  DINOv2-with-registers-base (frozen)
CLS token (768-dim)
    ↓  VAE-SIGREG Encoder (frozen)
z vector (3072-dim)  ← mu + σε
    ↓  cond_proj + cond_gate (학습)
DiT time-conditioning stream
```

**구현 세부 사항** (`ddt_vae.py`):
- `cond_proj`: Linear(3072 → encoder_hidden_size=1152)
- `cond_gate`: 학습 가능한 벡터 (0으로 초기화 → 사전 학습 가중치 보존)
- `null_cond`: 학습 가능한 null embedding (0으로 초기화, 3072-dim)
- 시간 임베딩에 합산: `c = t_embed + y_embed + cond_proj(z) × cond_gate`
- Classifier-free guidance: 학습 시 10% 확률로 z를 `null_cond`로 교체, 추론 시 uncond path도 동일하게 `null_cond` 사용

### Config 파일

| Config | Stage 1 체크포인트 | 비고 |
|---|---|---|
| `DiTDH-XL_DINOv2-B_VAESIGREG.yaml` | `200epoch.pt` (기본) | cond_dim=0 (비활성) |
| `DiTDH-XL_DINOv2-B_VAESIGREG_1e3.yaml` | `1e-3_200epoch.pt` | full ckpt 저장 |
| `DiTDH-XL_DINOv2-B_VAESIGREG_3e5.yaml` | `3e-5_200epoch.pt` | ema only 저장 |

### 학습 설정

| 항목 | 값 |
|---|---|
| Stage 1 인코더 | DINOv2-with-registers-base |
| Stage 2 모델 | DiTDH-XL (hidden [1152, 2048], depth [28, 2]) |
| 샘플 온도 | 1.0 |
| 에폭 | 100 |
| 배치 사이즈 | 8 |
| 학습률 | 1e-4 (AdamW, β=(0.9, 0.95)) |
| EMA | 0.9995 |
| 체크포인트 | 5에폭마다, 최근 3개 보존 |

### 학습 실행

```bash
cd VAE-SIGREG-RAE

# 학습 (기본)
bash run_train.sh
# 또는
python src/train_vae_cond.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_VAESIGREG_1e3.yaml \
  --data-path /path/to/ffhq256/imagefolder/train \
  --results-dir ckpts \
  --image-size 256 \
  --precision bf16 \
  --workers 8
```

### 선택적 캐싱

```bash
# DINO CLS 특징 사전 캐싱
python src/cache_dino_cls.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_VAESIGREG.yaml \
  --data-path /path/to/ffhq256/imagefolder/train \
  --output-dir cache/ffhq256_dino_cls_aug2

# 캐시 활용 학습
python src/train_vae_cond.py \
  --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_VAESIGREG_1e3.yaml \
  --data-path /path/to/ffhq256/imagefolder/train \
  --cached-latents-dir cache/ffhq256_dino_cls_aug2 \
  --results-dir ckpts \
  --image-size 256 \
  --precision bf16
```

### 평가 스크립트

```bash
# Gaussian prior로 unconditional 생성 후 FID 측정
python src/eval_gaussian_fid.py \
    --ckpt ckpts/<run>/ep-0040.pt \
    --config configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B_VAESIGREG_1e3.yaml \
    --real-path /path/to/ffhq256/imagefolder/val \
    --num-samples 5000 \
    --batch-size 16 \
    --precision bf16
```

---

## 전체 파이프라인 요약

```
[Stage 1] FFHQ256 DINO CLS 임베딩 (63,001장)
         → Overcomplete VAE 학습 (SIGReg 정규화)
         → 인수분해된 잠재 공간 z ∈ ℝ^3072 획득

[Stage 2] 이미지 + frozen DINOv2 + frozen VAE 인코더
         → z 벡터를 zero-gated adapter로 DiT-DH 파인튜닝
         → 조건부 이미지 생성 (RAE 디코더 출력)
```

---

## 핵심 의존성

```
torch >= 2.0
torchvision
transformers     # Dinov2WithRegistersModel
omegaconf        # YAML 설정 파싱 (Stage 2)
wandb            # 학습 로깅
torch-fidelity   # FID 평가 (Stage 2)
tqdm             # 진행 표시 (선택)
```

---

## 주요 파일 목록

| 파일 | 역할 |
|---|---|
| `VAE-SIGREG-SAE/src/vae_sigreg/model.py` | `OvercompleteVariationalAE` 아키텍처 |
| `VAE-SIGREG-SAE/src/vae_sigreg/losses.py` | `reconstruction_loss`, `kl_bottleneck_loss`, `sigreg_loss` |
| `VAE-SIGREG-SAE/src/vae_sigreg/diagnostics.py` | `compute_latent_diagnostics` (6종 잠재 통계) |
| `VAE-SIGREG-SAE/src/vae_sigreg/sample.py` | `sample_from_prior` |
| `VAE-SIGREG-SAE/train_run.py` | Stage 1 학습 진입점 |
| `VAE-SIGREG-RAE/src/vae_rae/conditioning.py` | `VaeSigregConditioner` 클래스 |
| `VAE-SIGREG-RAE/src/vae_rae/models/ddt_vae.py` | `DiTwDDTHeadVAECond` 모델 |
| `VAE-SIGREG-RAE/src/train_vae_cond.py` | Stage 2 학습 진입점 |
| `VAE-SIGREG-RAE/src/eval_gaussian_fid.py` | Gaussian prior 샘플링 → FID 평가 |
| `VAE-SIGREG-RAE/reconstruct_10x.py` | 10× 재구성 시각화 |
