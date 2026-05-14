# IGAE (Independent Gaussian Autoencoder)

DINOv2 비전 임베딩 위에서 **SIGReg(Sliced Isotropic Gaussian Regularization)** 정규화를 적용한 초과완전(Overcomplete) VAE를 학습하고, 그 잠재 표현을 조건으로 활용해 조건부 이미지 생성 모델(RAE + DiT)을 파인튜닝하는 2단계 생성 모델 파이프라인입니다.

---

## 프로젝트 구조

```
IGAE/
├── stage1_IGAE/              # Stage 1: Overcomplete VAE with SIGReg
└── stage2_conditioned_RAE/   # Stage 2: Conditional RAE (DiT-DH fine-tuning)
```

---

## Stage 1: IGAE

### 개요

DINOv2 CLS 임베딩(768-dim)을 입력으로 받아 3072-dim 잠재 공간으로 매핑하는 초과완전 VAE입니다. KL divergence 외에 **SIGReg** 정규화를 추가해 집합 사후 분포(aggregate posterior) `q(z)`가 등방성 가우시안 `N(0, I)`에 가까워지도록 강제합니다.

### 아키텍처

| 구성 요소 | 세부 내용 |
|---|---|
| 입력 | DINOv2 CLS 임베딩 (768-dim) |
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
| `L_KL` | 샘플별 KL divergence (KL annealing: 50에폭에 걸쳐 0 → β_kl 선형 증가) |
| `L_SIGReg` | 랜덤 단위벡터로 z를 1D 투영 후 경험적 특성함수(ECF)를 N(0,1) CF와 매칭 |

**SIGReg 상세** (Epps-Pulley): 랜덤 단위벡터로 z를 1D 슬라이스로 투영한 뒤, 각 슬라이스의 경험적 특성함수(ECF)를 N(0,1) 특성함수 `exp(-t²/2)`와 비교합니다.

- t ∈ [-5, 5] 구간의 33개 주파수 포인트 사용
- ECF 실수부/허수부 오차의 가중합 최소화 (N(0,1) CF 크기로 가중)
- z를 표준화하지 않고 직접 비교 → 평균·분산·고차 모멘트 동시 교정

### 학습 설정

| 하이퍼파라미터 | 값 |
|---|---|
| 데이터셋 | FFHQ256 DINOv2 CLS 임베딩 (63,001 train) |
| 에폭 | 200 |
| 배치 사이즈 | 512 (그래디언트 누적 4× → 유효 2048) |
| 학습률 | 3e-4 (코사인 감쇠 + 5에폭 웜업) |
| optimizer | AdamW (β1=0.9, β2=0.95, weight_decay=1e-2) |
| 투영 수 | 3072 |

### 학습 실행

```bash
cd stage1_IGAE
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

`best_ckpt/` 에 200 epoch active-survival sweep 기준 최종 체크포인트가 저장됩니다.

| 파일 | β_kl | λ_sigreg | Rec ↓ | KL | Active@0.1 ↑ | Active@0.5 ↑ | 비고 |
|---|---|---|---|---|---|---|---|
| `beta_1e-4_lambda_3_best.pt` | 1e-4 | 3 | 0.092 | 172.7 | 3072 | 3022 | 최저 Rec, 전체 차원 활성 |
| `beta_3e-4_lambda_3_best.pt` | 3e-4 | 3 | 0.109 | 95.3 | 3059 | 81 | Rec/KL 균형 |
| `beta_1e-3_lambda_10_best.pt` | 1e-3 | 10 | 0.140 | 48.8 | 3072 | 205 | 가장 낮은 KL |

```python
ckpt = torch.load("best_ckpt/beta_1e-3_lambda_10_best.pt", map_location="cpu", weights_only=False)
# ckpt["model"], ckpt["epoch"], ckpt["args"]
model = OvercompleteVariationalAE(**{k: ckpt["args"][k] for k in ["input_dim", "latent_dim", "hidden_dim", "num_layers"]})
model.load_state_dict(ckpt["model"])
```

### 분석 및 시각화 스크립트

`stage1_IGAE/scripts/` 에서 실행합니다.

| 스크립트 | 설명 |
|---|---|
| `preprocess_ffhq_dino.py` | FFHQ256에서 DINOv2 CLS 임베딩 추출 |
| `exp_active_units.py` | 차원별 KL 분석 → 활성 잠재 차원 식별 |
| `visualize_neuron.py` | 뉴런별 극단 이미지 그리드 (4×4) |
| `visualize_covariance.py` | 공분산 히트맵 + 인수분해 메트릭 |
| `plot_sigreg_sweeps.py` | SIGReg ablation 곡선 플롯 |
| `plot_sigreg_scurve_mpl.py` | active units S-curve 시각화 |
| `plot_stage5_paper_figures.py` | 최종 논문 피규어 생성 |
| `plot_total_correlation.py` | 전체 상관관계 히트맵 |

### 진단 메트릭 (`diagnostics.py`)

- 차원별 평균 절댓값
- 분산 오차 (목표: 1)
- 비대각 공분산 크기
- 왜도 및 초과 첨도

---

## Stage 2: Conditioned RAE

### 개요

Stage 1에서 학습한 IGAE의 **잠재 벡터(3072-dim)** 를 조건 신호로 사용해 DiT-DH(Diffusion Transformer with DDT Head) 기반 RAE를 파인튜닝합니다. 이미지 → DINOv2 → 고정된 IGAE 인코더 → z = mu + σε → DiT 조건 주입의 흐름으로 동작합니다.

### 조건화 메커니즘

```
Image (256×256)
    ↓  DINOv2-with-registers-base (frozen)
CLS token (768-dim)
    ↓  IGAE Encoder (frozen)
z vector (3072-dim)  ← mu + σε
    ↓  cond_proj + cond_gate (학습)
DiT time-conditioning stream
```

**구현 세부 사항** (`ddt_vae.py`):
- `cond_proj`: Linear(3072 → encoder_hidden_size=1152)
- `cond_gate`: 학습 가능한 벡터 (0으로 초기화 → 사전 학습 가중치 보존)
- `null_cond`: 학습 가능한 null embedding (0으로 초기화, 3072-dim)
- 시간 임베딩에 합산: `c = t_embed + y_embed + cond_proj(z) × cond_gate`
- Classifier-free guidance: 학습 시 10% 확률로 z를 `null_cond`로 교체

### Config 파일

| Config | Stage 1 체크포인트 | 비고 |
|---|---|---|
| `DiTDH-XL_DINOv2-B_VAESIGREG.yaml` | — | cond_dim=0 (비활성, 베이스라인) |
| `DiTDH-XL_DINOv2-B_VAESIGREG_1e3.yaml` | `beta_1e-3_lambda_10_best.pt` | full ckpt 저장 |
| `DiTDH-XL_DINOv2-B_VAESIGREG_3e5.yaml` | `beta_1e-4_lambda_3_best.pt` | ema only 저장 |

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
cd stage2_conditioned_RAE
bash run_train.sh
# 또는 직접 실행
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
[Stage 1] FFHQ256 DINOv2 CLS 임베딩 (63,001장)
         → Overcomplete VAE 학습 (SIGReg 정규화)
         → 인수분해된 잠재 공간 z ∈ ℝ^3072 획득

[Stage 2] 이미지 + frozen DINOv2 + frozen IGAE 인코더
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
| `stage1_IGAE/src/vae_sigreg/model.py` | `OvercompleteVariationalAE` 아키텍처 |
| `stage1_IGAE/src/vae_sigreg/losses.py` | `reconstruction_loss`, `kl_bottleneck_loss`, `EppsPulleySIGReg` |
| `stage1_IGAE/src/vae_sigreg/diagnostics.py` | `compute_latent_diagnostics` (6종 잠재 통계) |
| `stage1_IGAE/src/vae_sigreg/sample.py` | `sample_from_prior` |
| `stage1_IGAE/train_run.py` | Stage 1 학습 진입점 |
| `stage2_conditioned_RAE/src/vae_rae/conditioning.py` | `VaeSigregConditioner` 클래스 |
| `stage2_conditioned_RAE/src/vae_rae/models/ddt_vae.py` | `DiTwDDTHeadVAECond` 모델 |
| `stage2_conditioned_RAE/src/train_vae_cond.py` | Stage 2 학습 진입점 |
| `stage2_conditioned_RAE/src/eval_gaussian_fid.py` | Gaussian prior 샘플링 → FID 평가 |
| `stage2_conditioned_RAE/reconstruct_10x.py` | 10× 재구성 시각화 |
