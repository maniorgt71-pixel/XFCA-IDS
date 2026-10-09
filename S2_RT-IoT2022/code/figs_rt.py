"""
figs_rt.py - Figures 11-13 of the manuscript from results_rtiot/results.json (output of rt_iot2022.py).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results_rtiot/results.json"))
BLUE, ORANGE = "#1f77b4", "#ff7f0e"
LAB = ("Hybrid FFNN+XGBoost", "Baseline (raw XGBoost)")

# Figure 11: per-category F1, five-class task
rows = R["E1"]["5class"]
base = np.array([r["base"]["per_class"] for r in rows]); hyb = np.array([r["hyb"]["per_class"] for r in rows])
cats = ["Benign", "DoS/DDoS", "Recon.", "BruteForce\n(n=37)", "Spoofing/\nMITM"]
x = np.arange(5); w = 0.36
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(x - w / 2, hyb.mean(0), w, yerr=hyb.std(0, ddof=1), capsize=3, color=BLUE, label=LAB[0])
ax.bar(x + w / 2, base.mean(0), w, yerr=base.std(0, ddof=1), capsize=3, color=ORANGE, label=LAB[1])
ax.set_xticks(x, cats); ax.set_ylim(0.6, 1.02); ax.set_ylabel("F1 (mean ± SD, 5 seeds)")
ax.set_title("RT-IoT2022: per-class F1, 5-class task"); ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig("figure11_perclass_f1.png", dpi=150)

# Figure 12: detection of withheld subtypes
W = [("NMAP_XMAS_TREE_SCAN", "XMAS tree scan\n(n=2,010)"), ("NMAP_FIN_SCAN", "FIN scan\n(n=28)"),
     ("DDOS_Slowloris", "Slowloris\n(n=534)"), ("Metasploit_Brute_Force_SSH", "Brute-force SSH\n(n=37)")]
hb = np.array([[e[k]["hyb"] for k, _ in W] for e in R["E3"]]); bb = np.array([[e[k]["base"] for k, _ in W] for e in R["E3"]])
x = np.arange(len(W))
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(x - w / 2, hb.mean(0), w, yerr=hb.std(0, ddof=1), capsize=3, color=BLUE, label=LAB[0])
ax.bar(x + w / 2, bb.mean(0), w, yerr=bb.std(0, ddof=1), capsize=3, color=ORANGE, label=LAB[1])
ax.set_xticks(x, [l for _, l in W]); ax.set_ylim(0, 1.28); ax.set_ylabel("Detection rate on unseen subtype")
ax.set_title("RT-IoT2022: detection of attack subtypes withheld from training")
ax.legend(loc="upper center", ncol=2); fig.tight_layout(); fig.savefig("figure12_unseen_subtypes.png", dpi=150)

# Figure 13: TreeSHAP top ten, five-class baseline (seed 42)
top = R["E2"]["ranking"][:10][::-1]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh([n for n, _ in top], [v for _, v in top], color="#4c72b0")
ax.set_xlabel("mean |SHAP value| (averaged over 5 classes)")
ax.set_title("TreeSHAP feature attribution (RT-IoT2022, baseline XGBoost)")
fig.tight_layout(); fig.savefig("figure13_treeshap.png", dpi=150)
print("Figures 11-13 written.")
