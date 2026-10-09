# S2 - RT-IoT2022 validation (Sections 4.1.4, 4.2.5 and 4.6; Algorithm 3)

## Code
- `code/rt_iot2022.py` - experiments E1-E4 of Algorithm 3 (hybrid FFNN + XGBoost vs. raw XGBoost, five seeds 42-46).
  Without options it runs the protocol of Tables 13-16; with `--dedup` it runs the duplicate-free protocol of
  Section 4.6.6 (port-free feature vectors that occur with more than one label are removed, and every remaining
  distinct vector is kept once).
- `code/rt_duplicates.py` - duplicate structure of the release (Sections 4.1.4 and 4.6.6): duplicates with and
  without ports, distinct feature vectors per label, share of test records with an exact copy in the training split,
  and overlap of the withheld E3 subtypes with the retained records.
- `code/figs_rt.py` - Figures 11-13 from `results_rtiot/results.json`.

## Reproduce (from inside `results/`)
```
python ../code/rt_iot2022.py            # -> results_rtiot/results.json        (Tables 13-17, Figures 11-13)
python ../code/rt_iot2022.py --dedup    # -> results_rtiot_dedup/results.json  (Section 4.6.6, Table 17)
python ../code/rt_duplicates.py         # -> results_rtiot_duplicates.json     (Sections 4.1.4 and 4.6.6)
python ../code/figs_rt.py               # -> figure11-13 *.png
```
`rt_iot2022.py` downloads RT_IOT2022.csv (github.com/Binwakil/RT_IOT2022_IDS; original: UCI Machine Learning
Repository, doi:10.24432/C5P338) into the working directory and checks its SHA-256 checksum (Table 24).
Runtime: about 10 min (original protocol) and 2 min (duplicate-free) on one CPU core.

## results/
`results_rtiot/results.json` and `results_rtiot_dedup/results.json` (per-seed metrics for E1 and E3, TreeSHAP ranking
and stability for E2, size and latency for E4, Wilcoxon summaries), `results_rtiot_duplicates.json`, the console
logs of the three runs, and Figures 11-13.

Inference latency (E4) depends on the load of the machine; on the shared single-vCPU environment used here, repeated
runs of the same script gave different absolute values, so only its order of magnitude should be compared.

## Environment
Python 3, NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, SciPy 1.17.1, XGBoost 3.4.1, SHAP 0.52.0, matplotlib;
single-vCPU Intel Xeon 2.80 GHz, 3 GB RAM.
