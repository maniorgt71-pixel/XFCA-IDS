import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stats import M, S, DOM, RESULTS

# Figure 14: held-out macro-F1 by held-out domain, both schedules
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
panels = [(axs[0], M, [("within", "Within-domain reference"), ("RAW", "Raw XGBoost"), ("BASE", "Baseline hybrid (λ = 0)"),
                       ("FIXED", "Fixed-λ hybrid (λ = 0.3)"), ("XFCA", "XFCA (adaptive λ)")], "(a) E = 90, τ = 0.6 (main setting)"),
          (axs[1], S, [("within", "Within-domain reference"), ("RAW", "Raw XGBoost"), ("BASE", "Baseline hybrid (λ = 0)"),
                       ("FIXED", "Fixed-λ hybrid (λ = 0.3)"), ("RAMP", "SHAP-blind λ ramp"), ("XFCA", "XFCA (adaptive λ)")], "(b) E = 300, τ = 0.95 (sensitivity)")]
colors = {"RAW": "tab:blue", "BASE": "tab:orange", "FIXED": "tab:green", "RAMP": "tab:purple", "XFCA": "tab:red"}
for ax, R, conds, title in panels:
    n = len(conds); w = 0.8 / n
    for j, (k, lab) in enumerate(conds):
        mu, sd = [], []
        for d in DOM:
            v = [r["within_f1"] if k == "within" else r[k]["f1"] for r in R if r["held"] == d]
            mu.append(np.mean(v)); sd.append(np.std(v, ddof=1))
        x = np.arange(3) + (j - (n - 1) / 2) * w
        if k == "within":
            ax.bar(x, mu, w, yerr=sd, capsize=2, label=lab, color="lightgrey", hatch="//", edgecolor="grey")
        else:
            ax.bar(x, mu, w, yerr=sd, capsize=2, label=lab, color=colors[k])
    ax.set_xticks(np.arange(3)); ax.set_xticklabels([f"{d}\n(held out)" for d in DOM])
    ax.set_title(title); ax.set_ylim(0, 1.05)
axs[0].set_ylabel("Macro-F1 (mean ± SD, 5 seeds)")
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(RESULTS / "fig14.png", dpi=150); plt.close(fig)

# Figure 15: lambda and checkpoint rho trajectories
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
for R, lab, tau, c in [(M, "E = 90, τ = 0.6", 0.6, "tab:blue"), (S, "E = 300, τ = 0.95", 0.95, "tab:red")]:
    ep = [0] + [l["epoch"] for l in R[0]["XFCA"]["log"]]
    lam = np.array([[0.3] + [l["lam_after"] for l in r["XFCA"]["log"]] for r in R])
    axs[0].errorbar(ep, lam.mean(0), yerr=lam.std(0, ddof=1), marker="o", ms=3, capsize=2, color=c, label=f"XFCA, {lab}")
    ep2 = [l["epoch"] for l in R[0]["XFCA"]["log"]]
    rx = np.array([[l["rho"] for l in r["XFCA"]["log"]] for r in R])
    rb = np.array([[l["rho"] for l in r["BASE"]["log"]] for r in R])
    axs[1].plot(ep2, rx.mean(0), marker="o", ms=3, color=c, label=f"XFCA, {lab}")
    axs[1].plot(ep2, rb.mean(0), marker="s", ms=3, ls="--", color=c, alpha=0.6, label=f"Baseline, {lab}")
    axs[1].axhline(tau, color=c, lw=0.8, ls=":")
ramp = [0] + [l["epoch"] for l in S[0]["RAMP"]["log"]]
axs[0].plot(ramp, [0.3] + [l["lam_after"] for l in S[0]["RAMP"]["log"]], ls="--", color="tab:purple", label="SHAP-blind ramp, E = 300")
axs[0].axhline(0.3, color="tab:green", lw=1, ls=":", label="Fixed λ = 0.3")
axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Adversarial weight λ (mean ± SD)")
axs[0].set_title("(a) λ after each SHAP checkpoint"); axs[0].legend(fontsize=7)
axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Checkpoint SHAP-consistency ρ (mean)")
axs[1].set_title("(b) Checkpoint ρ (dotted lines: target τ)"); axs[1].legend(fontsize=7, loc="lower left")
axs[1].set_ylim(0.4, 1.0)
fig.tight_layout(); fig.savefig(RESULTS / "fig15.png", dpi=150); plt.close(fig)
print("ok")
