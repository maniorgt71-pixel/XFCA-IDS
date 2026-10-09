# Supplementary File S3 — Complete XFCA on real cross-dataset domains (Section 4.7)

## Data (unmodified public releases; place in `data/` next to `code/`, or set the environment variable XFCA_DATA)

| File | Records | SHA-256 | Source used |
|---|---|---|---|
| KDDTrain+.txt | 125973 | `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95` | github.com/defcom17/NSL_KDD |
| KDDTest+.txt | 22544 | `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84` | github.com/defcom17/NSL_KDD |
| UNSW_NB15_training-set.csv | 175341 | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` | github.com/Nir-J/ML-Projects (UNSW-Network_Packet_Classification) |
| UNSW_NB15_testing-set.csv | 82332 | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` | github.com/Nir-J/ML-Projects (UNSW-Network_Packet_Classification) |
| RT_IOT2022.csv | 123117 | `d9b315d6d899cd60d356f7d3dbe61907f1163985bd800a87cffe600eb577bd20` | github.com/Binwakil/RT_IOT2022_IDS (Data/); original: UCI ML Repository, doi:10.24432/C5P338 |

## Environment
Python 3.12.3, NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, SciPy 1.17.1, XGBoost 3.4.1, SHAP 0.52.0, matplotlib;
single-vCPU Intel Xeon 2.80 GHz, 3 GB RAM (same environment as Section 4.2.5).

## Code
- `code/harmonize.py` — maps the three releases to the shared 20-feature schema and 3-class label space (Tables 18–19).
- `code/xfca_real.py` — full Algorithm 1 (NumPy FFNN encoder + GRL domain branch + TreeSHAP consistency control of lambda + XGBoost) under leave-one-domain-out, 5 seeds; conditions RAW, BASE, FIXED, XFCA, and the SHAP-blind RAMP control.
- `code/stats.py` — all numbers in Section 4.7 and Tables 20–22 (run: `python stats.py`).
- `code/figs.py` — Figures 14 and 15.

## Reproduce
```
python xfca_real.py 90 0.6                              # main setting       -> results_E90_tau0.6.json
python xfca_real.py 300 0.95                            # sensitivity        -> results_E300_tau0.95.json
MODES=RAMP ONLY_MODES=1 python xfca_real.py 300 0.95    # SHAP-blind ramp    -> results_E300_tau0.95_RAMP.json
python stats.py; python figs.py
```
All scripts resolve paths relative to the bundle: data are read from `data/` (override with XFCA_DATA) and results are
written to and read from `results/` (override with XFCA_RESULTS). Run the commands from inside `code/`.
Runtime on the environment above: about 11 min (main), 27 min (sensitivity) and 7 min (ramp).

## Settings fixed by the authors (not in Table 7)
Adam, learning rate 0.01, full batch; delta = 0.1, lambda_min = 0, lambda_max = 1; top-m lists compared over their union, absent feature rank = m + 1;
TreeSHAP on 2,000 random records per source domain; 15,000 stratified records per domain per seed; held-out domain split 70/30.

## results/
Per-fold JSON (seed, held-out domain, within-domain F1, per-condition macro-F1, accuracy, per-class F1, final rho, domain-probe accuracy,
decision-layer size, checkpoint log of rho and lambda) and the console logs of the three runs.

## Revision notes (this version)
- Paths: the scripts no longer assume `/home/claude/`; see *Reproduce* above.
- `stats.py`: a change of lambda at a checkpoint is counted only if it exceeds 1e-9. The earlier strict comparison counted
  floating-point rounding steps (e.g. 0.9999999999999999 -> 1.0) as updates. Corrected counts (Table 22): XFCA, E = 300,
  tau = 0.95: 111 increases / 13 decreases of 150 checkpoints (26 further checkpoints had rho < tau with lambda already at
  lambda_max = 1); SHAP-blind ramp: 105 increases / 0 decreases of 150. Main setting unchanged (2 / 43 of 45).
  No other reported number changes.
