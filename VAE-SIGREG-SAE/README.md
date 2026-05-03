# VAE-SIGREG-SAE

Overcomplete Variational Autoencoder with SIGReg regularization, trained on DINO CLS embeddings of FFHQ256.

## 개요

입력 차원(768)보다 큰 잠재 공간(3072)을 사용하는 **Overcomplete VAE**.
KL loss 외에 **SIGReg(Sliced Isotropic Gaussian Regularization)** 를 추가해 aggregate posterior q(z)를 N(0,I)로 밀어 co-activation 없는 독립적 잠재 표현을 학습.

```
Loss = Reconstruction (MSE) + β_kl(t) · KL + λ_sigreg · SIGReg
```

## 모델 구조

```
Input (768) → Encoder MLP → (mu, logvar) → z ~ N(mu, exp(logvar)) → Decoder MLP → Output (768)
```

- Encoder/Decoder: 대칭 MLP, LayerNorm + GELU
- `num_layers=4`, `hidden_dim=768×4=3072`, `latent_dim=3072`
- Reparameterization trick으로 샘플링

## Loss 상세

### KL Divergence (`beta_kl`)
```
KL( N(μ, σ²) || N(0,1) ) = 0.5 * (μ² + σ² - ln(σ²) - 1)
```
- 가우시안 closed-form이라 적분 없이 계산
- 학습 초반 posterior collapse 방지를 위해 0 → β_kl 로 annealing

### SIGReg (`lambda_sigreg`)
랜덤 방향 벡터로 z를 1D로 projection한 뒤, 해당 분포의 moment를 N(0,1)에 맞춤:
```
L_sig = mean² + (var-1)² + 0.1·skew² + 0.1·(kurt-3)²
```
- KL은 per-sample posterior를 규제, SIGReg는 aggregate posterior q(z)를 규제
- `num_projections=3072` 개의 랜덤 방향 사용

## 학습 실행

```bash
bash run_train.sh
```

주요 하이퍼파라미터 (`run_train.sh` 기준):

| 파라미터 | 값 |
|---|---|
| `input_dim` | 768 |
| `latent_dim` | 3072 |
| `epochs` | 200 |
| `batch_size` | 512 × 4 (accum) = 2048 유효 배치 |
| `lr` | 3e-4 (cosine decay, warmup 5 epoch) |
| `beta_kl` | 1e-2 (50 epoch annealing) |
| `lambda_sigreg` | 0.05 |
| `num_projections` | 3072 |

데이터: `FFHQ256` DINO CLS features (train 63,001 / eval 별도)

## 스크립트

| 스크립트 | 설명 |
|---|---|
| `train_run.py` | 학습 메인 |
| `run_train.sh` | 학습 실행 래퍼 (nohup, 로그 저장) |
| `exp_active_units.py` | 차원별 KL로 활성 뉴런 수 분석 |
| `exp_interpolation.py` | 잠재 공간 보간 시각화 |
| `visualize_neuron.py` | KL 기준 상위 뉴런의 대표 이미지 시각화 |
| `visualize_latent.py` | 특정 차원의 극단값 이미지 시각화 |
| `visualize_dist.py` | 뉴런별 mu 분포 히스토그램 |
| `visualize_covariance.py` | 잠재 공간 상관행렬 및 분산 분석 |

## 시각화 결과 (`visualizations/`)

| 파일 | 내용 |
|---|---|
| `covariance_z.png` | 상관행렬 heatmap + off-diagonal 분포 + 차원별 분산 |
| `active_units.png` | 뉴런별 mean KL + 활성 뉴런 누적 곡선 |
| `interpolation.png` | 이미지 쌍 간 잠재 공간 보간 |
| `neuron_distributions.png` | 상위 뉴런의 mu 분포 |
| `neurons_v2/`, `neurons_v3/` | 뉴런별 극단 이미지 그리드 |

## 체크포인트

`checkpoints/` 에 학습 완료된 체크포인트가 저장됩니다.

| 파일 | β_kl | SIGReg |
|---|---|---|
| `1e-2_200epoch.pt` | 1e-2 | ✓ |
| `1e-3_200epoch.pt` | 1e-3 | ✓ |
| `3e-5_200epoch.pt` | 3e-5 | ✓ |
| `3e-5_200_noSIG_epoch.pt` | 3e-5 | ✗ (ablation) |

```python
ckpt = torch.load("checkpoints/1e-3_200epoch.pt", map_location="cpu", weights_only=False)
# ckpt["model"], ckpt["epoch"], ckpt["args"]
model = OvercompleteVariationalAE(**{k: ckpt["args"][k] for k in ["input_dim", "latent_dim", "hidden_dim", "num_layers"]})
model.load_state_dict(ckpt["model"])
```
