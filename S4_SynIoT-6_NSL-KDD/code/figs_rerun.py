import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("XFCA_RESULTS", ROOT / "results"))
DATA = os.environ.get("XFCA_DATA", str(ROOT / "data"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11})
GREY, BLUE = "#808080", "#1f4e79"
SY = json.load(open(RESULTS / "results_syniot6.json")); R = SY["results"]; DOMS = list(SY["params"])
NS = json.load(open(RESULTS / "results_nslkdd.json"))
labels = ["D1\nCICIoT2023", "D2\nTON_IoT", "D3\nN-BaIoT", "D4\nBoT-IoT", "D5\nIoT-23", "D6\nEdge-IIoTset"]
OUT = str(RESULTS / "figs_rerun")
import os; os.makedirs(OUT, exist_ok=True)


def per_dom(mode, key):
    mu, sd = [], []
    for d in DOMS:
        rows = [r for r in R if r["held"] == d]
        v = [r["within_f1"] - r[mode]["f1"] for r in rows] if key == "gap" else [r[mode]["f1"] for r in rows]
        mu.append(np.mean(v)); sd.append(np.std(v, ddof=1))
    return mu, sd


for key, fname, ylab, title in [("f1", "image4.png", "Held-out macro-F1", "Held-out (cross-domain) macro-F1 by domain (mean ± SD, 3 seeds)"),
                                ("gap", "image5.png", "Generalization gap (within − held-out)", "Generalization gap by domain (lower is better; mean ± SD, 3 seeds)")]:
    fig, ax = plt.subplots(figsize=(9.6, 5.07))
    x = np.arange(6); w = 0.36
    for j, (m, lab, c) in enumerate([("BASE", "Baseline hybrid", GREY), ("XFCA", "XFCA (proposed)", BLUE)]):
        mu, sd = per_dom(m, key)
        ax.bar(x + (j - 0.5) * w, mu, w, yerr=sd, capsize=4, color=c, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_ylabel(ylab); ax.set_title(title, fontsize=11)
    if key == "f1": ax.set_ylim(0, 1.18)
    ax.legend(frameon=False, ncol=2, loc="upper right"); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=150); plt.close(fig)

# Figure 6: overall summary
vals = {m: [np.mean([r[m]["f1"] for r in R]), np.mean([r["within_f1"] - r[m]["f1"] for r in R]),
            np.mean([r[m]["rho_final"] for r in R])] for m in ("BASE", "XFCA")}
fig, ax = plt.subplots(figsize=(8.53, 4.8))
x = np.arange(3); w = 0.36
for j, (m, lab, c) in enumerate([("BASE", "Baseline hybrid", GREY), ("XFCA", "XFCA (proposed)", BLUE)]):
    b = ax.bar(x + (j - 0.5) * w, vals[m], w, color=c, label=lab)
    for rect, v in zip(b, vals[m]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(["Held-out macro-F1", "Generalization gap\n(lower is better)", "SHAP top-8\nconsistency (ρ)"])
ax.set_ylim(0, 1.0); ax.set_title("SynIoT-6 overall summary (6 domains × 3 seeds)", fontsize=12)
ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/image6.png", dpi=150); plt.close(fig)

# Figure 7: NSL-KDD class distribution
C = ["Benign", "DoS/DDoS", "Reconnaissance", "BruteForce", "WebInject"]
tr = [NS["counts"]["train"][c] for c in C]; te = [NS["counts"]["test"][c] for c in C]
fig, ax = plt.subplots(figsize=(8.8, 4.8)); x = np.arange(5); w = 0.36
ax.bar(x - w / 2, tr, w, color=GREY, label=f"KDDTrain+ (n = {sum(tr):,})")
ax.bar(x + w / 2, te, w, color=BLUE, label=f"KDDTest+ (n = {sum(te):,})")
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(C); ax.set_ylabel("Records (log scale)")
ax.set_title("NSL-KDD class distribution: training vs. test split", fontsize=12)
ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/image7.png", dpi=150); plt.close(fig)

# Figure 8: NSL-KDD per-class F1 and overall metrics (mean ± SD over 5 seeds)
NR = NS["results"]
def agg(m):
    pc = np.array([r[m]["pcf1"] for r in NR]); mf = [r[m]["macro_f1"] for r in NR]; ac = [r[m]["acc"] for r in NR]
    return list(pc.mean(0)) + [np.mean(mf), np.mean(ac)], list(pc.std(0, ddof=1)) + [np.std(mf, ddof=1), np.std(ac, ddof=1)]
fig, ax = plt.subplots(figsize=(9.6, 4.93)); x = np.arange(7); w = 0.36
for j, (m, lab, c) in enumerate([("BASE", "Baseline hybrid", GREY), ("DA", "XFCA (domain-adversarial only)", BLUE)]):
    mu, sd = agg(m)
    b = ax.bar(x + (j - 0.5) * w, mu, w, yerr=sd, capsize=3, color=c, label=lab)
    for rect, v, s in zip(b, mu, sd):
        ax.text(rect.get_x() + rect.get_width() / 2, v + s + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
ax.axvline(4.5, ls="--", color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(C + ["Macro-F1", "Accuracy"], fontsize=9); ax.set_ylabel("Score (mean ± SD, 5 seeds)")
ax.set_ylim(0, 1.05); ax.set_title("NSL-KDD validation: per-class F1 and overall metrics", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper center"); ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f"{OUT}/image8.png", dpi=150); plt.close(fig)
print("ok")
