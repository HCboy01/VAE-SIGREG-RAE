# IGAE Stage-1: KL Loss Variants

Overcomplete VAE with linear decoder on DINO features (FFHQ-256).  
The central question: **어떤 KL loss가 소수의 dim만 선택적으로 활성화하면서 reconstruction도 잘 되게 하는가?**

---

## 모델 / 데이터

| 항목 | 값 |
|------|----|
| 데이터 | DINO ViT-B/8 features, FFHQ-256 (63001 × 768) |
| 모델 | OvercompleteVariationalAE, linear decoder |
| latent_dim | 6144 (input_dim × 8) |
| num_layers | 4 |
| β warmup | linear ramp, 50 epochs |
| sigreg | Epps-Pulley (num_projections=512) |
| epochs | 200, batch_size=2048, lr=3e-4 |

---

## KL Loss 변형 실험

### 배경: 이상적인 latent space란?

- **대부분의 dim**: μ≈0, σ≈1 (prior와 동일, 비활성)
- **소수의 dim**: μ≠0, σ<1 (특정 개념을 인코딩, 활성)
- σ>1 또는 σ→0은 모두 비정상적

---

### 1. `feature` — Feature-level KL (기준선)

```
L = Σ_d [ (mean_n[μ_d])²  +  (mean_n[σ²_d] - 1)² ]
```

**의도**: sample KL은 always-on feature(모든 이미지에서 활성화된 dim)와 selective feature(일부 이미지에서만 활성)를 구분하지 못함.
Feature KL은 배치 평균 μ가 작은 selective dim에는 gradient가 작아 → reconstruction이 dominant → feature가 살아남는 selectivity 메커니즘을 가짐.

**문제**: σ² 항을 평균 낸 뒤 제곱하므로, 어떤 이미지에서 σ>1, 다른 이미지에서 σ<1이면 상쇄되어 σ>1이 허용됨.

| run | rec | active@0.1 | σ>1 | 특이사항 |
|-----|-----|-----------|-----|---------|
| b1e-3, λ=10 | 0.087 | 6144 | 있음 | 전 dim 약하게 활성 |
| b1e-4, λ=10 | 0.004 | 407 | 있음 | 405개 강한 bottleneck, 우수한 marginal (std≈1, kurt≈0) |
| b1e-3, λ=100 | 0.169 | 6144 | 있음 | sigreg가 σ 억제, kl_feature 작음 |
| b1e-4, λ=100 | 0.061 | 407 | 있음 | b1e-4 l10보다 활성 dim 적음 |

---

### 2. `feature_sq` — Squared Per-dim Mean KL

```
L = Σ_d ( mean_n[μ²_d] + mean_n[σ²_d - logσ²_d - 1] )²
  = 4 · Σ_d (mean_n[KL_d])²
```

**의도**: feature KL의 σ 상쇄 문제를 해결하기 위해 μ²와 σ²-logσ²-1을 각각 per-sample로 계산한 뒤 평균(부호 상쇄 없음), 그 합을 제곱.
Gradient ∝ mean_n[KL_d] · μ_{n,d} → 낮은 KL dim은 gradient가 작아 selectivity 보존 기대.

**문제**: Σ_d (mean_n[KL_d])²는 KL이 dim들에 **고르게 퍼질 때 최솟값** (볼록함수). 결과적으로 모든 dim이 비슷하게 약한 KL을 가지도록 수렴 → reconstruction 크게 저하.

| run | rec | σ>1 | 특이사항 |
|-----|-----|-----|---------|
| b1e-3, λ=10 | 0.180 | 없음 | sigma 전 dim 균일(≈0.87) |
| b1e-3, λ=100 | 0.213 | 없음 | 거의 collapse |
| b1e-4, λ=10 | 0.109 | 없음 | sigma 균일, KL 분산 없음 |
| b1e-4, λ=100 | 0.155 | 없음 | 4개 dim만 σ<0.8 |

---

### 3. `feature_log` v1 — Log Penalty (μ+σ 통합)

```
L = Σ_d log(1 + mean_n[μ²_d] + mean_n[σ²_d - logσ²_d - 1])
  = Σ_d log(1 + 2·mean_n[KL_d])
```

**의도**: 오목함수(concave) 패널티로 sparsity 유도.
Σ_d log(1+f_d)는 f_d가 집중될 때 최솟값 → 소수 dim에 KL 집중을 선호.
Gradient ∝ μ_{n,d} / (1 + f_d) → 활성 dim(f_d 큰)은 gradient 포화 → 살아남음.

**문제**: log 안에 σ 항이 함께 있어 f_d가 이미 크면 σ→0이어도 penalty 증가가 미미 → σ→0 허용 (deterministic AE 경향).

| run | rec | active dim | σ<0.2 | 특이사항 |
|-----|-----|-----------|-------|---------|
| b1e-3, λ=10 | 0.228 | ≈0 | 0 | 거의 collapse |
| b1e-3, λ=100 | 0.251 | 28 | - | 극소수만 활성 |
| b1e-4, λ=10 | 0.047 | 417 | **235** | sparsity 작동, σ→0 문제 |
| b1e-4, λ=100 | 0.107 | 185 | **140** | λ높을수록 더 sparse |

---

### 4. `feature_log` v2 — Log Penalty (μ/σ 분리)

```
L = Σ_d log(1 + mean_n[μ²_d])           ← μ: log penalty로 sparsity
  + Σ_d mean_n[σ²_d - logσ²_d - 1]      ← σ: log 밖에 직접 배치
```

**의도**: v1의 σ→0 문제는 σ 항이 log 안에 있어 포화되었기 때문.
σ 항을 log 밖으로 분리하면 σ→0 시 `-logσ²→∞`가 직접 작동 → σ→0 방지.
동시에 μ의 log penalty는 sparsity를 유지.

| 항 | 역할 |
|----|------|
| `log(1 + mean_n[μ²_d])` | 소수 dim만 μ로 인코딩 (sparsity) |
| `mean_n[σ²_d - logσ²_d - 1]` | σ∈(0,∞) 범위에서 1로 유도, 양 극단 방지 |

| run | rec | active@0.1 | active@0.5 | kl_total | z_fd_diag | 특이사항 |
|-----|-----|-----------|-----------|---------|---------|---------|
| b1e-3, λ=10 | 0.087 | 6144 | 7 | 2325.7 | 348.09 | 전 dim 활성, KL 폭발 |
| b1e-3, λ=100 | 0.169 | 6144 | 0 | 1340.8 | 88.17 | 전 dim 활성, KL 폭발 |
| b1e-4, λ=10 | 0.004 | 5696 | 1774 | 8814.2 | 756.44 | KL 극도 폭발, prior 완전 이탈 |
| b1e-4, λ=100 | 0.061 | 407 | **405** | 2160.8 | 66.84 | σ→0 방지 확인, sparsity 작동 |

**문제**: β=1e-4, λ=10처럼 SIGReg가 약할 때 KL이 폭발적으로 증가 (kl_total=8814). log penalty가 큰 KL dim에 gradient를 포화시켜 KL이 무제한 성장함. λ=100으로 SIGReg를 강하게 걸었을 때만 407 dim의 양호한 sparsity 달성.

---

### 5. `sample_l1` — Sample KL + L1 on |μ|

```
L = 0.5 · mean_n[ Σ_d (μ²_{n,d} + σ²_{n,d} - logσ²_{n,d} - 1) ]   ← 표준 sample KL
  + α · mean_n mean_d |μ_{n,d}|                                       ← Laplace prior
```

**의도**: 표준 sample KL의 μ gradient는 ∝ μ (L2) → μ가 작으면 gradient도 작아 "약하게 살아있는" dim이 많이 잔존.
L1 추가 시 gradient = ±α (상수) → μ가 작아도 일정 크기로 0 방향 push → 더 날카로운 sparsity 기대 (Lasso 효과).

| run | rec | active@0.1 | kl_total | feat_never_active@0.1 | 특이사항 |
|-----|-----|-----------|---------|----------------------|---------|
| b1e-3, α=0 (baseline) | 0.177 | 41 | 80.8 | 0.714 | |
| b1e-3, α=0.1 | 0.204 | 49 | 82.5 | 0.846 | α 효과 미미 |
| b1e-3, α=1.0 | 0.204 | 49 | 82.3 | 0.857 | α=0.1과 동일 |
| b1e-4, α=0 (baseline) | 0.108 | 386 | 526.1 | 0.139 | |
| b1e-4, α=0.1 | 0.059 | 348 | 494.0 | **0.652** | dead dim 크게 증가 |
| b1e-4, α=1.0 | 0.059 | 348 | 494.0 | **0.586** | α=0.1과 동일 |

**문제**: α를 10배 높여도 (0.1→1.0) 결과가 동일 → L1이 약한 dim을 죽이는 임계점에 α=0.1에서 이미 도달. 이후 살아남는 dim들은 reconstruction gradient가 압도적으로 커서 α를 높여도 추가 억제 불가.

더 근본적으로, L1은 sample KL과 마찬가지로 **배치 통계를 보지 않는** per-sample 패널티이므로, always-on dim과 selective dim을 구분하는 메커니즘이 없음. feature KL 계열이 이 문제를 더 직접적으로 해결함.

---

## KL Loss 설계 원칙 (누적 학습)

| 문제 | 원인 | 해결 방향 |
|------|------|----------|
| σ>1 허용 | `(mean_n[σ²_d]-1)²`: 평균 후 제곱 → 상쇄 가능 | σ 항은 per-sample 계산 후 평균 |
| KL 균일 확산 | L2(squared) on per-dim KL: 볼록함수, spread 선호 | 오목함수(log) 사용 |
| σ→0 허용 | σ 항이 log 안에 있어 포화 | σ 항은 log 밖에 독립 배치 |
| selectivity 부재 | gradient가 모든 dim에 동일 크기 | 배치 통계(mean_n[KL_d])로 gradient 스케일 조정 |
| KL 폭발 | log penalty가 큰 KL dim에 gradient 포화 → 무제한 성장 | SIGReg(λ)로 외부 억제 필요, 또는 KL upper bound 도입 |
| L1 α 불감도 | 임계점 이하 dim은 α=0.1로도 충분히 억제 / 이상 dim은 recon이 압도 | α 탐색보다 배치 통계 기반 설계가 근본 해결책 |

---

## 구현 위치

| 파일 | 내용 |
|------|------|
| `src/vae_sigreg/losses.py` | `kl_feature_level`, `kl_feature_level_sq`, `kl_feature_level_log`, `l1_mu_loss` |
| `train_run.py` | `--kl_type {sample, feature, feature_sq, feature_log, sample_l1}`, `--alpha_l1` |
| `scripts/sweep_feat_kl.sh` | feature KL sweep (β × λ) |
| `scripts/sweep_featklsq.sh` | feature_sq sweep |
| `scripts/sweep_featkllog.sh` | feature_log sweep (v1→v2 공용) |
| `scripts/sweep_samplel1.sh` | sample_l1 sweep (β × α_l1) |
| `scripts/batch_mu_sigma_combined_lindec.py` | 시각화: μ-σ scatter (KL quantile 5패널 + per-dim scatter) |
| `visualizations/mu_sigma_combined_lindec/` | 각 checkpoint별 시각화 결과 PNG |
