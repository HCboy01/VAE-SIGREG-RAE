# VAE-SIGREG-SAE Ablation Plan

현재 목표는 `latent_dim=3072`을 고정하고, 각 `beta_kl`마다 active latent가 급격히 줄어드는 지점 직전의 `lambda_sigreg`를 찾는 것이다.

## Selection Rule

active neuron 기준 correlation 분포를 볼 예정이므로 `val/active_units`가 가장 중요하다. SIGReg sweep에서는 `lambda_sigreg`를 키우다가 `val/active_units`가 cliff처럼 급락하기 직전의 가장 큰 값을 해당 `beta_kl`의 후보로 고른다. 단, `val/rec_loss`가 크게 망가지면 제외한다.

## Key Results So Far

### Stage 1. KL Sweep, `latent_dim=3072`, `lambda_sigreg=0.05`

| `beta_kl` | rec | KL | SIG | total | var err | offdiag | active | 선택 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3e-5 | 0.0936 | 590.4 | 9.53e-5 | 0.1114 | 0.0467 | 0.0275 | 3072 | 보류 |
| 1e-4 | 0.1154 | 266.9 | 7.74e-5 | 0.1421 | 0.0340 | 0.0217 | 3072 | 보류 |
| 3e-4 | 0.1451 | 132.1 | 7.25e-5 | 0.1848 | 0.0266 | 0.0194 | 3072 | 후보 |
| 1e-3 | 0.1946 | 58.6 | 6.94e-5 | 0.2532 | 0.0254 | 0.0183 | 2900 | 후보 |
| 3e-3 | 0.2593 | 26.2 | 6.70e-5 | 0.3381 | 0.0247 | 0.0179 | 777 | 보류 |
| 1e-2 | 0.3722 | 8.8 | 6.86e-5 | 0.4599 | 0.0245 | 0.0177 | 1 | 제외 |
| 3e-2 | 0.5155 | 1.5 | 6.83e-5 | 0.5616 | 0.0252 | 0.0177 | 0 | 제외 |

### Stage 2. SIGReg Sweep at `beta_kl=1e-3`, `latent_dim=3072`

| `lambda_sigreg` | rec | KL | SIG | total | var err | offdiag | active | 선택 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.1948 | 58.4 | 0 | 0.2531 | 0.0256 | 0.0184 | 2900 | 참고 |
| 0.005 | 0.1934 | 58.9 | 6.82e-5 | 0.2523 | 0.0242 | 0.0183 | 2916 | 후보 |
| 0.05 | 0.1937 | 58.8 | 6.88e-5 | 0.2525 | 0.0249 | 0.0180 | 2917 | 후보 |
| 0.1 | 0.1932 | 57.8 | 6.88e-5 | 0.2509 | 0.0248 | 0.0181 | 2856 | 후보 |
| 0.5 | 0.1944 | 58.7 | 6.87e-5 | 0.2531 | 0.0250 | 0.0182 | 2926 | 선택 |

### Stage 3. Latent Sweep at `beta_kl=1e-3`, `lambda_sigreg=0.5`

| `latent_dim` | rec | KL | SIG | total | var err | offdiag | active | 선택 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1536 | 0.1591 | 51.8 | 6.78e-5 | 0.2109 | 0.0321 | 0.0179 | 542 | 제외 |
| 3072 | 0.1935 | 59.0 | 6.98e-5 | 0.2525 | 0.0251 | 0.0182 | 2927 | 선택 |
| 6144 | 0.1991 | 57.5 | 6.81e-5 | 0.2567 | 0.0247 | 0.0178 | 2153 | 보류 |

## Stage 4. Per-KL SIGReg Sweep

`latent_dim=3072` 고정. 각 `beta_kl`마다 SIGReg grid를 돌려 active cliff 직전 `lambda_sigreg`를 찾는다.

기본 grid:

```text
0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0
```

### `beta_kl=3e-4`

2026-05-04 14:08 실행은 불완전했고, 2026-05-04 14:28 retry로 `0.01`부터 `1.0`까지 완료했다.

| `lambda_sigreg` | rec | KL | SIG | total | var err | offdiag | active | 선택 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.1461 | 131.7 | 0 | 0.1856 |  |  |  | 참고 |
| 0.01 | 0.1445 | 132.3 | 7.09e-5 | 0.1842 | 0.0274 | 0.0202 | 3071.5 | 후보 |
| 0.03 | 0.1455 | 132.3 | 7.05e-5 | 0.1852 | 0.0267 | 0.0197 | 3072 | 후보 |
| 0.05 | 0.1451 | 132.5 | 7.12e-5 | 0.1848 | 0.0264 | 0.0196 | 3072 | 후보 |
| 0.1 | 0.1455 | 130.6 | 7.10e-5 | 0.1847 | 0.0274 | 0.0195 | 3072 | 후보 |
| 0.2 | 0.1440 | 132.3 | 7.15e-5 | 0.1837 | 0.0275 | 0.0191 | 3072 | 후보 |
| 0.5 | 0.1450 | 131.8 | 7.05e-5 | 0.1845 | 0.0269 | 0.0202 | 3072 | 후보 |
| 1.0 | 0.1433 | 132.1 | 7.14e-5 | 0.1830 | 0.0278 | 0.0192 | 3070.5 | 후보 |

판단: `lambda_sigreg=1.0`까지 active cliff가 오지 않았다. `0.2`와 `1.0`이 total/offdiag 측면에서 좋아 보이지만, 목표는 active latent 급락 지점이므로 더 큰 SIGReg 영역을 추가로 본다.

다음 실행은 같은 `beta_kl=3e-4`에서 high-SIG range를 확인한다.

```bash
bash launch_next_experiment.sh
```

현재 next 설정:

```text
beta_kl=3e-4
latent_dim=3072
SIG_GRID=2 5 10 20 50 100 200 500
WANDB_INIT_TIMEOUT=300
LAUNCH_SLEEP=5
```

이 high-SIG sweep이 끝나면 `beta_kl=3e-4`의 active cliff 직전 `lambda_sigreg`를 기록한다. cliff가 그래도 안 오면 다음 beta는 `1e-3`, 그 다음은 `3e-3` 순서로 진행한다.

2026-05-04 14:39 high-SIG sweep 완료.

| `lambda_sigreg` | rec | KL | SIG | total | var err | offdiag | active | 선택 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.1431 | 132.8 | 7.08e-5 | 0.1831 | 0.0267 | 0.0197 | 3071 | 후보 |
| 5 | 0.1431 | 133.0 | 7.21e-5 | 0.1834 | 0.0265 | 0.0195 | 3072 | 후보 |
| 10 | 0.1423 | 131.8 | 7.10e-5 | 0.1826 | 0.0284 | 0.0191 | 3070.5 | 후보 |
| 20 | 0.1473 | 126.5 | 7.05e-5 | 0.1867 | 0.0294 | 0.0189 | 3063 | 선택 |
| 50 | 0.1616 | 115.6 | 7.03e-5 | 0.1998 | 0.0303 | 0.0188 | 2894 | cliff 시작 |
| 100 | 0.1752 | 98.4 | 6.98e-5 | 0.2117 | 0.0305 | 0.0187 | 1940.75 | active 급락 |
| 200 | 0.1971 | 77.0 | 6.92e-5 | 0.2340 | 0.0296 | 0.0178 | 120.75 | collapse |
| 500 | 0.2523 | 67.2 | 6.81e-5 | 0.3065 | 0.0280 | 0.0175 | 64 | collapse |

결론: `beta_kl=3e-4`의 `lambda_sigreg` 후보는 `20`이다. `50`부터 active가 줄기 시작하고, `100`에서 급격히 꺾인다.

다음 실행은 `beta_kl=1e-3`에서 같은 방식으로 active cliff를 찾는다.

```text
beta_kl=1e-3
latent_dim=3072
SIG_GRID=1 2 5 10 20 50 100 200
```

### `beta_kl=1e-3`

2026-05-04 14:51 sweep 완료.

| `lambda_sigreg` | rec | KL | SIG | total | var err | offdiag | active | 선택 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.1940 | 58.6 | 6.80e-5 | 0.2527 | 0.0253 | 0.0183 | 2925 | 후보 |
| 2 | 0.1925 | 59.4 | 6.83e-5 | 0.2520 | 0.0252 | 0.0182 | 2940.25 | 후보 |
| 5 | 0.1909 | 59.5 | 6.91e-5 | 0.2507 | 0.0249 | 0.0182 | 2899 | 후보 |
| 10 | 0.1901 | 59.7 | 7.00e-5 | 0.2504 | 0.0247 | 0.0183 | 2942.75 | 후보 |
| 20 | 0.1943 | 58.7 | 6.93e-5 | 0.2544 | 0.0250 | 0.0185 | 2904.75 | 선택 |
| 50 | 0.2051 | 55.9 | 6.94e-5 | 0.2645 | 0.0252 | 0.0187 | 2651 | cliff 시작 |
| 100 | 0.2170 | 53.3 | 7.07e-5 | 0.2774 | 0.0252 | 0.0182 | 2413.75 | active 감소 |
| 200 | 0.2393 | 48.9 | 6.93e-5 | 0.3021 | 0.0249 | 0.0188 | 2028 | active 감소 |

결론: `beta_kl=1e-3`의 `lambda_sigreg` 후보는 `20`이다. `50`부터 active가 분명히 줄기 시작한다. `10`은 active/total이 가장 좋으므로 보조 후보로 기록한다.

다음 실행은 `beta_kl=3e-3`에서 낮은 SIGReg 범위를 다시 본다. Stage 1에서 `beta=3e-3, lambda=0.05`일 때 active가 이미 777까지 줄었으므로 high-SIG가 아니라 낮은 lambda 주변을 확인한다.

```text
beta_kl=3e-3
latent_dim=3072
SIG_GRID=0 0.001 0.003 0.005 0.01 0.02 0.03 0.05
```

### `beta_kl=3e-3`

2026-05-04 15:01 sweep 완료.

| `lambda_sigreg` | rec | KL | SIG | total | var err | offdiag | active | 선택 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.2601 | 26.2 | 0 | 0.3388 | 0.0252 | 0.0181 | 786 | 참고 |
| 0.001 | 0.2593 | 26.3 | 6.86e-5 | 0.3383 | 0.0251 | 0.0179 | 798.75 | 후보 |
| 0.003 | 0.2594 | 26.2 | 6.81e-5 | 0.3381 | 0.0253 | 0.0178 | 721.75 | active 감소 |
| 0.005 | 0.2592 | 26.3 | 6.72e-5 | 0.3380 | 0.0249 | 0.0180 | 783.75 | 후보 |
| 0.01 | 0.2597 | 26.2 | 6.91e-5 | 0.3385 | 0.0249 | 0.0178 | 758 | active 감소 |
| 0.02 | 0.2590 | 26.2 | 6.79e-5 | 0.3377 | 0.0248 | 0.0180 | 767.75 | active 감소 |
| 0.03 | 0.2592 | 26.3 | 6.94e-5 | 0.3381 | 0.0250 | 0.0175 | 776.25 | active 감소 |
| 0.05 | 0.2594 | 26.2 | 6.86e-5 | 0.3379 | 0.0250 | 0.0180 | 728.25 | active 감소 |

결론: `beta_kl=3e-3`는 `lambda_sigreg=0`에서도 active가 786으로 이미 낮다. active 기준으로 이 beta 자체를 보류/제외한다. 굳이 후보를 남기면 active가 가장 높은 `lambda_sigreg=0.001`이지만, cliff 탐색 목적에는 적합하지 않다.

다음 실행은 낮은 KL 쪽 `beta_kl=1e-4`에서 high-SIG range를 확인한다.

```text
beta_kl=1e-4
latent_dim=3072
SIG_GRID=10 20 50 100 200 500 1000 2000
```

### `beta_kl=1e-4`

2026-05-04 15:25 sweep 완료. `lambda=20`은 W&B init timeout으로 실패했다.

| `lambda_sigreg` | rec | KL | SIG | total | var err | offdiag | active | 선택 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0.1159 | 256.6 | 7.36e-5 | 0.1423 | 0.0380 | 0.0215 | 3072 | 후보 |
| 20 |  |  |  |  |  |  |  | 실패 |
| 50 | 0.1417 | 198.2 | 7.28e-5 | 0.1651 | 0.0367 | 0.0191 | 2122 | cliff 이후 |
| 100 | 0.1640 | 179.4 | 7.08e-5 | 0.1890 | 0.0335 | 0.0190 | 575.5 | active 급락 |
| 200 | 0.1919 | 161.7 | 7.04e-5 | 0.2221 | 0.0341 | 0.0182 | 253.5 | collapse |
| 500 | 0.2600 | 134.7 | 7.01e-5 | 0.3085 | 0.0439 | 0.0178 | 172.5 | collapse |
| 1000 | 0.2771 | 141.4 | 6.86e-5 | 0.3598 | 0.0566 | 0.0180 | 183.25 | collapse |
| 2000 | 0.2975 | 156.8 | 6.85e-5 | 0.4501 | 0.0705 | 0.0180 | 147 | collapse |

판단: `10`에서는 active가 3072로 유지되고, `50`에서는 이미 active가 2122까지 감소했다. cliff는 `10`과 `50` 사이에 있다. `lambda=20` 실패 때문에 후보를 확정하지 않고 중간 구간을 재실험한다.

다음 실행은 `beta_kl=1e-4`에서 `15~45` 구간을 확인한다.

```text
beta_kl=1e-4
latent_dim=3072
SIG_GRID=15 20 25 30 35 40 45
WANDB_INIT_TIMEOUT=600
LAUNCH_SLEEP=10
```

2026-05-04 15:46 refinement sweep 완료.

| `lambda_sigreg` | rec | KL | SIG | total | var err | offdiag | active | 선택 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 15 | 0.1193 | 248.3 | 7.50e-5 | 0.1453 | 0.0382 | 0.0206 | 3071.5 | 후보 |
| 20 | 0.1232 | 235.3 | 7.44e-5 | 0.1482 | 0.0402 | 0.0194 | 3070.25 | 후보 |
| 25 | 0.1261 | 228.3 | 7.34e-5 | 0.1508 | 0.0394 | 0.0191 | 3062.75 | 후보 |
| 30 | 0.1298 | 221.9 | 7.33e-5 | 0.1542 | 0.0390 | 0.0189 | 3005.5 | 후보 |
| 35 | 0.1311 | 212.3 | 7.22e-5 | 0.1548 | 0.0385 | 0.0196 | 2924 | 선택 |
| 40 | 0.1348 | 209.4 | 7.13e-5 | 0.1586 | 0.0376 | 0.0198 | 2761 | cliff 시작 |
| 45 | 0.1386 | 205.2 | 7.25e-5 | 0.1623 | 0.0375 | 0.0191 | 2447.75 | active 급락 |

결론: `beta_kl=1e-4`의 `lambda_sigreg` 후보는 `35`이다. `40`부터 active가 분명히 줄기 시작한다. `30`은 active/rec 보조 후보로 기록한다.

## Figures

SIGReg sweep figure는 아래 스크립트로 재생성한다. 현재 repository에는 50 epoch 기반 figure 산출물을 남기지 않는다. 200 epoch shared-grid sweep이 끝나면 같은 스크립트의 `DATA`를 새 결과로 갱신해서 다시 그린다.

```bash
cd /root/workspace/VAE-SIGREG-RAE/VAE-SIGREG-SAE
/root/workspace/miniconda3/envs/vae-sigreg-sae/bin/python scripts/plot_sigreg_sweeps.py
/root/workspace/miniconda3/envs/vae-sigreg-sae/bin/python scripts/plot_sigreg_scurve_mpl.py
```

Figure 해석: `lambda_sigreg`를 키우면 off-diagonal covariance error는 대체로 낮아지며 Gaussianity가 개선되지만, 일정 지점 이후 active units가 급격히 줄어 latent collapse가 시작된다. 따라서 후보는 covariance가 개선된 상태에서 active cliff 직전에 있는 값으로 고른다.

## Final Candidate Retrain

50 epoch 기준 최종 후보:

| 후보 | `beta_kl` | `lambda_sigreg` | 이유 |
|---|---:|---:|---|
| `final_b1e_4_sig35` | 1e-4 | 35 | 낮은 KL에서 active cliff 직전 |
| `final_b3e_4_sig20` | 3e-4 | 20 | active 유지와 KL 균형 |
| `final_b1e_3_sig20` | 1e-3 | 20 | active cliff 직전 |
| `final_b1e_3_sig10` | 1e-3 | 10 | active/total 보조 후보 |

다음 실행은 위 후보들을 200 epoch로 재학습한다.

### Final 200 Epoch Result

2026-05-04 16:06 실행 완료. `best.pt`는 `val/total_loss` 기준으로 저장된다.

| 후보 | best epoch | rec | KL | SIG | total | var err | offdiag | active | 판단 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `final_b1e_4_sig35` | 98 | 0.1243 | 139.6 | 6.77e-5 | 0.1407 | 0.0304 | 0.0202 | 74 | active collapse |
| `final_b3e_4_sig20` | 112 | 0.1160 | 90.6 | 6.85e-5 | 0.1446 | 0.0264 | 0.0201 | 54 | active collapse |
| `final_b1e_3_sig20` | 194 | 0.1430 | 48.1 | 6.75e-5 | 0.1925 | 0.0259 | 0.0199 | 76.75 | active collapse |
| `final_b1e_3_sig10` | 195 | 0.1453 | 49.3 | 6.85e-5 | 0.1952 | 0.0262 | 0.0202 | 213 | active collapse, 그나마 최고 |

판단: 50 epoch sweep에서는 active cliff 직전 후보가 잘 보였지만, 200 epoch까지 길게 돌리면 `val/total_loss`와 reconstruction은 좋아지는 대신 active units가 대부분 collapse한다. active neuron correlation 분석이 목적이면 이 200 epoch checkpoint를 최종으로 쓰면 안 된다. 다음 실험은 `best.pt` 저장 기준에 active 유지 조건을 넣거나, 50 epoch 후보를 기준으로 짧은 재현 실험을 여러 seed로 돌리는 쪽이 맞다.

## Stage 5. 200 Epoch Active-Survival Sweep

목표: 각 `beta_kl`마다 200 epoch 끝까지 dead neuron이 늘어나지 않는 `lambda_sigreg` 범위를 다시 측정한다. 이번부터 stdout에 매 epoch `ever_active`, `dead`, `kl_active`, `var`, `offdiag`를 같이 출력한다.

Dead neuron 기준: validation set의 모든 이미지에 대해 특정 latent dimension의 deterministic encoder activation `mu`가 한 번도 `abs(mu) > threshold`를 만족하지 않으면 dead로 본다. 즉 `ever_active_units = D - dead_units`이다. 기존 `active_units`는 `KL_d > 0.01` 기준이므로 `active_units_kl_0_01`로 보조 기록한다.

Dead threshold grid:

```text
0.1 0.5
```

W&B에는 예를 들어 아래처럼 threshold별 metric이 기록된다.

```text
val/ever_active_units_thr_0_1
val/dead_units_thr_0_1
val/dead_fraction_thr_0_1
val/ever_active_units_thr_0_5
val/dead_units_thr_0_5
val/dead_fraction_thr_0_5
```

`val/ever_active_units`, `val/dead_units`, `val/dead_fraction` alias는 checkpoint용 primary threshold `0.1` 기준이다.

저장 기준도 변경했다. Stage 5 스크립트는 `--best_metric active_safe_total`을 사용해서 `threshold=0.1` 기준 `val/ever_active_units >= MIN_ACTIVE_UNITS`인 epoch 중 `val/total_loss`가 가장 좋은 checkpoint만 `best.pt`로 저장한다. 조건을 만족하는 epoch가 없으면 해당 run은 `best.pt`가 생기지 않는다.

공통 lambda grid:

```text
0 1 3 10 30 100 300 1000
```

실행 순서:

| 순서 | script | `beta_kl` | `SIG_GRID` | `MIN_ACTIVE_UNITS` | 목적 |
|---:|---|---:|---|---:|---|
| 1 | `launch_stage5_b1e_4_200ep.sh` | 1e-4 | `0 1 3 10 30 100 300 1000` | 2500 | shared-grid 200 epoch S-curve |
| 2 | `launch_stage5_b3e_4_200ep.sh` | 3e-4 | `0 1 3 10 30 100 300 1000` | 2500 | shared-grid 200 epoch S-curve |
| 3 | `launch_stage5_b1e_3_200ep.sh` | 1e-3 | `0 1 3 10 30 100 300 1000` | 2500 | shared-grid 200 epoch S-curve |

`beta_kl=3e-3`는 main figure에서는 제외한다. 필요하면 `launch_stage5_b3e_3_200ep.sh`로 같은 grid를 돌릴 수 있다.

### Stage 5 Summary Table

아래 표는 모든 200 epoch shared-grid run이 끝나면 채운다. `Active@0.1`은 validation 전체 이미지에서 `abs(mu) > 0.1`을 한 번이라도 넘은 latent dimension 수이고, `Active@0.5`는 `abs(mu) > 0.5` 기준이다.

| `beta_kl` | `lambda_sigreg` | Regime | Rec ↓ | KL | Offdiag ↓ | Active@0.1 ↑ | Active@0.5 ↑ |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1e-4 | 3 | Best | 0.0918 | 172.7 | 0.0201 | 3072 | 3022 |
| 1e-4 | 30 | Stronger SIGReg | 0.1198 | 141.8 | 0.0199 | 3072 | 2831 |
| 1e-4 | 100 | Active@0.5 drop | 0.1657 | 125.5 | 0.0199 | 3072 | 2385 |
| 3e-4 | 3 | Best | 0.1088 | 95.3 | 0.0200 | 3059 | 81 |
| 3e-4 | 10 | Active@0.1 stable | 0.1119 | 93.0 | 0.0200 | 3070 | 107 |
| 3e-4 | 100 | Collapse | 0.1581 | 64.6 | 0.0199 | 32 | 28 |
| 1e-3 | 10 | Best | 0.1401 | 48.8 | 0.0201 | 3072 | 205 |
| 1e-3 | 100 | Active@0.5 drop | 0.1536 | 44.0 | 0.0201 | 2987 | 38 |
| 1e-3 | 1000 | Collapse | 0.2620 | 22.1 | 0.0201 | 10 | 8 |

참고: 예전 50 epoch 표의 `Low / Pre-cliff / Collapse` label은 새 200 epoch 결과에서 active cliff 위치를 보고 다시 확정한다.

각 `beta_kl`별 선택 checkpoint:

| `beta_kl` | selected `lambda_sigreg` | 이유 | copied checkpoint |
|---:|---:|---|---|
| 1e-4 | 3 | 가장 낮은 Rec/total, Active@0.1 전체 유지, Active@0.5 3022 유지 | `best_ckpt/beta_1e-4_lambda_3_best.pt` |
| 3e-4 | 3 | 가장 낮은 Rec/total, Active@0.1 3059 유지 | `best_ckpt/beta_3e-4_lambda_3_best.pt` |
| 1e-3 | 10 | 가장 낮은 Rec/total, Active@0.1 전체 유지 | `best_ckpt/beta_1e-3_lambda_10_best.pt` |

다음 실행:

```bash
cd /root/workspace/VAE-SIGREG-RAE/VAE-SIGREG-SAE
bash launch_next_experiment.sh
```

## Commands

```bash
cd /root/workspace/VAE-SIGREG-RAE/VAE-SIGREG-SAE
bash launch_next_experiment.sh
nvidia-smi
```
