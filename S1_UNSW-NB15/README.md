# S1 - UNSW-NB15 validation (Section 4.5, Algorithm 2)

Run:  python run_unsw_nb15.py

The script downloads the official UNSW-NB15 training and testing partitions, checks their
SHA-256 checksums (Table 24), runs five seeds (42-46) and writes its outputs to results_unsw/.
Runtime: about 5-10 minutes on a single CPU core.

Outputs of the run reported in the manuscript (included here):
- table11_summary.csv - Table 11
- table12_shap_ranking.csv - Table 12 (all 39 features)
- per_seed_results.csv - per-seed metrics
- figure9_metrics.png, figure10_shap.png - Figures 9 and 10
- run_config.json - features, seeds and settings used
