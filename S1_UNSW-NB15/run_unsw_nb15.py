"""
UNSW-NB15 validation of the hybrid FFNN-encoder + XGBoost pipeline (Algorithm 2),
rerun on the OFFICIAL UNSW-NB15 training and testing partitions.

Supplementary File S1 of the manuscript (Section 4.5, Algorithm 2, Tables 11-12, Figures 9-10).

Usage:
    pip install numpy pandas scikit-learn scipy xgboost shap matplotlib
    python run_unsw_nb15.py

The script downloads the two official CSV files (if not already present),
verifies their SHA-256 checksums against Table 24 of the manuscript, runs
five seeds (42-46), and writes all results to the folder 'results_unsw/'.
"""
import hashlib, json, os, pickle, urllib.request
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "https://raw.githubusercontent.com/Nir-J/ML-Projects/HEAD/UNSW-Network_Packet_Classification/"
FILES = {  # file name -> SHA-256 (Table 24)
    "UNSW_NB15_training-set.csv": "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
    "UNSW_NB15_testing-set.csv":  "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
}
SEEDS = [42, 43, 44, 45, 46]
RAW_SUBSET = ["sttl", "rate", "sload"]          # x_S in Eq. (7), as in Algorithm 2
XGB_PARAMS = dict(n_estimators=150, max_depth=6, learning_rate=0.15,
                  eval_metric="logloss", n_jobs=1)
OUT = "results_unsw"
os.makedirs(OUT, exist_ok=True)


def get_file(name, sha):
    if not os.path.exists(name):
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(BASE + name, name)
    h = hashlib.sha256(open(name, "rb").read()).hexdigest()
    if h != sha:
        raise SystemExit(f"Checksum mismatch for {name}: {h}")
    print(f"{name}: checksum OK")
    return pd.read_csv(name)


def embed(mlp, X):
    """Manual 2-layer forward pass, Eqs. (1)-(2): 16-dimensional embedding."""
    h = np.maximum(0, X @ mlp.coefs_[0] + mlp.intercepts_[0])
    return np.maximum(0, h @ mlp.coefs_[1] + mlp.intercepts_[1])


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred),
                recall=recall_score(y, pred), f1=f1_score(y, pred),
                mae=float(np.mean(np.abs(prob - y))))


train = get_file("UNSW_NB15_training-set.csv", FILES["UNSW_NB15_training-set.csv"])
test = get_file("UNSW_NB15_testing-set.csv", FILES["UNSW_NB15_testing-set.csv"])

# Numeric flow-level features only (drop row id, labels and the 3 categorical columns)
FEATS = [c for c in train.columns
         if c not in ("id", "label", "attack_cat") and pd.api.types.is_numeric_dtype(train[c])]
Xtr, ytr = train[FEATS].to_numpy(float), train["label"].to_numpy()
Xte, yte = test[FEATS].to_numpy(float), test["label"].to_numpy()
print(f"Train {Xtr.shape}, test {Xte.shape}, {len(FEATS)} numeric features")

scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
idx_S = [FEATS.index(f) for f in RAW_SUBSET]

rows, baseline_models = [], {}
for seed in SEEDS:
    base = xgb.XGBClassifier(random_state=seed, **XGB_PARAMS).fit(Xtr, ytr)
    baseline_models[seed] = base
    mb = metrics(yte, base.predict_proba(Xte)[:, 1])

    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                        early_stopping=True, max_iter=300, random_state=seed).fit(Xtr_s, ytr)
    Htr = np.hstack([embed(mlp, Xtr_s), Xtr[:, idx_S]])
    Hte = np.hstack([embed(mlp, Xte_s), Xte[:, idx_S]])
    hyb = xgb.XGBClassifier(random_state=seed, **XGB_PARAMS).fit(Htr, ytr)
    mh = metrics(yte, hyb.predict_proba(Hte)[:, 1])

    for k in mb:
        rows.append(dict(seed=seed, metric=k, baseline=mb[k], hybrid=mh[k]))
    print(f"seed {seed}: baseline acc {mb['accuracy']:.4f} F1 {mb['f1']:.4f} | "
          f"hybrid acc {mh['accuracy']:.4f} F1 {mh['f1']:.4f}")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/per_seed_results.csv", index=False)

# Summary table (Table 11 replacement): mean ± SD, hybrid wins, Wilcoxon p
summary = []
for m in ["accuracy", "precision", "recall", "f1", "mae"]:
    d = df[df.metric == m]
    diff = d.hybrid.to_numpy() - d.baseline.to_numpy()
    better = (diff < 0) if m == "mae" else (diff > 0)
    p = wilcoxon(d.hybrid, d.baseline).pvalue if np.any(diff != 0) else 1.0
    summary.append(dict(metric=m,
        baseline=f"{d.baseline.mean():.4f} ± {d.baseline.std(ddof=1):.4f}",
        hybrid=f"{d.hybrid.mean():.4f} ± {d.hybrid.std(ddof=1):.4f}",
        delta=f"{diff.mean():+.4f}", hybrid_wins=int(better.sum()), wilcoxon_p=round(float(p), 4)))
summary = pd.DataFrame(summary)
summary.to_csv(f"{OUT}/table11_summary.csv", index=False)
print("\nTable 11 (official partitions, 5 seeds):\n", summary.to_string(index=False))

# TreeSHAP on the seed-42 baseline, 2,000-record test subsample (Table 12 / Figure 10)
rng = np.random.RandomState(42)
sub = Xte[rng.choice(len(Xte), 2000, replace=False)]
sv = shap.TreeExplainer(baseline_models[42]).shap_values(sub)
imp = pd.Series(np.abs(sv).mean(0), index=FEATS).sort_values(ascending=False)
imp.to_csv(f"{OUT}/table12_shap_ranking.csv", header=["mean_abs_shap"])
print("\nTable 12 (top 10 TreeSHAP features):\n", imp.head(10).round(4).to_string())

# Figures 9 and 10 (as in the manuscript)
ms = ["accuracy", "precision", "recall", "f1", "mae"]
x = np.arange(len(ms)); w = 0.35
fig, ax = plt.subplots(figsize=(7, 4.5))
for j, (col, lab, colr) in enumerate([("hybrid", "Hybrid FFNN+XGBoost", "#1f77b4"),
                                      ("baseline", "Baseline (raw XGBoost)", "#ff7f0e")]):
    g = df.groupby("metric")[col]
    ax.bar(x + (j - 0.5) * w, g.mean()[ms], w, yerr=g.std()[ms], capsize=3, label=lab, color=colr)
ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1", "MAE"]); ax.set_ylim(0, 1.18)
ax.set_title("UNSW-NB15 (official partitions, 5 seeds): Hybrid vs. Baseline XGBoost")
ax.legend(loc="upper center", ncol=2); fig.tight_layout()
fig.savefig(f"{OUT}/figure9_metrics.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4.5))
top = imp.head(14)[::-1]
ax.barh(top.index, top.values, color="#4c72b0"); ax.set_xlabel("mean |SHAP value|")
ax.set_title("TreeSHAP feature attribution (UNSW-NB15, XGBoost)"); fig.tight_layout()
fig.savefig(f"{OUT}/figure10_shap.png", dpi=150)

json.dump(dict(features=FEATS, n_train=len(Xtr), n_test=len(Xte), seeds=SEEDS,
               raw_subset=RAW_SUBSET, xgb_params=XGB_PARAMS),
          open(f"{OUT}/run_config.json", "w"), indent=2)
print(f"\nAll results written to '{OUT}/'.")
