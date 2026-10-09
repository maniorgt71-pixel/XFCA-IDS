# Supplementary File S4 — SynIoT-6 generator, XFCA implementation, and the SynIoT-6 and NSL-KDD experiments

## SynIoT-6 generator (code/syniot6.py, function generate)
Shared class signatures: mu_c ~ N(0, 1.5^2 I_16), drawn once (generator seed 2024).
Record of class c in domain k: x = a_k * (mu_c + eps) + b_k, eps ~ N(0, sigma_k^2 I), a_k ~ U(lo_k, hi_k) per feature, b_k ~ N(0, s_k^2) per feature.
1,400 records per domain (8,400 total). Classes: Benign, DoS/DDoS, Recon., BruteForce, Botnet, Web/Inj..

| Domain | Records per class (Benign / DoS/DDoS / Recon. / BruteForce / Botnet / Web/Inj.) | Noise sigma_k | Scale range [lo, hi] | Shift sd s_k |
|---|---|---|---|---|
| D1 (CICIoT2023-like) | 140 / 770 / 140 / 70 / 210 / 70 | 0.8 | [0.7, 1.3] (drawn 0.71–1.25) | 0.6 (drawn -1.07 to 0.24) |
| D2 (TON_IoT-like) | 490 / 280 / 210 / 140 / 70 / 210 | 0.9 | [0.7, 1.3] (drawn 0.74–1.24) | 0.8 (drawn -1.38 to 1.39) |
| D3 (N-BaIoT-like) | 210 / 420 / 140 / 70 / 532 / 28 | 1.0 | [0.6, 1.4] (drawn 0.69–1.23) | 0.8 (drawn -1.21 to 1.29) |
| D4 (BoT-IoT-like) | 28 / 980 / 280 / 28 / 42 / 42 | 1.2 | [0.5, 1.5] (drawn 0.54–1.20) | 1.2 (drawn -2.37 to 1.18) |
| D5 (IoT-23-like) | 350 / 210 / 350 / 70 / 378 / 42 | 1.1 | [0.6, 1.4] (drawn 0.74–1.39) | 1.0 (drawn -1.88 to 1.32) |
| D6 (Edge-IIoTset-like) | 560 / 280 / 168 / 112 / 70 / 210 | 1.0 | [0.6, 1.4] (drawn 0.62–1.34) | 1.0 (drawn -1.69 to 2.58) |

Sanity check (raw XGBoost, within-domain 70/30 split): `python syniot6.py check` gives macro-F1 between 0.90 (D4) and 0.98 (D1).
Standardization in the experiments: fitted on the pooled source domains of each fold, applied unchanged to the held-out domain.

## XFCA implementation (code/xfca_core.py)
NumPy full-batch FFNN encoder (32-16 ReLU), class head, domain head (16 ReLU), gradient reversal (Eq. 5), Adam (lr 0.01),
E = 90, Kc = 30, snapshot XGBoost 40 trees depth 4, final XGBoost 150 trees depth 6 lr 0.15, m = 8, lambda0 = 0.3, tau = 0.6,
delta = 0.1, [lambda_min, lambda_max] = [0, 1] (Eq. 10; decrease step delta/3). Eq. (9): top-m lists compared over their union,
absent feature rank m + 1. TreeSHAP on up to 2,000 records per source domain. Reported rho: final decision layer, source domains.

## Experiments
- `python syniot6.py` — LODO, 6 domains x seeds 42-44; conditions RAW, BASE (lambda = 0), FIXED (lambda = 0.3), XFCA;
  bootstrap resampling of each source domain per seed; all models scored on the same 30% of the held-out domain;
  within-domain reference trained on the other 70%. -> results/results_syniot6.json (Tables 8-9, Figures 4-6). ~10 min.
- `python nslkdd_da.py` — KDDTrain+ (labelled source) vs KDDTest+ (unlabelled target, domain branch only), seeds 42-46,
  standard 5-category mapping incl. test-only subtypes, 122 features (38 numeric with log1p on heavy-tailed counts + one-hot
  protocol/service/flag), standardization on KDDTrain+; conditions RAW, BASE, DA (lambda = 0.3 fixed). -> results/results_nslkdd.json
  (Table 10, Figures 7-8). ~10 min. Data files: see Supplementary File S3 (same NSL-KDD files and checksums).
- `python stats_rerun.py` prints every number used in Sections 4.3-4.4; `python figs_rerun.py` regenerates Figures 4-8.

## Environment
Python 3.12.3, NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, SciPy 1.17.1, XGBoost 3.4.1, SHAP 0.52.0, matplotlib;
single-vCPU Intel Xeon 2.80 GHz, 3 GB RAM.

## Paths (revision note)
All scripts resolve paths relative to the bundle: input data are read from `data/` next to `code/` (override with the
environment variable XFCA_DATA; NSL-KDD files as in Supplementary File S3), and results are written to and read from
`results/` (override with XFCA_RESULTS). Run the commands from inside `code/`. Earlier versions assumed `/home/claude/`.
