import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("XFCA_RESULTS", ROOT / "results"))
DATA = os.environ.get("XFCA_DATA", str(ROOT / "data"))
import numpy as np
from scipy.stats import wilcoxon, spearmanr

SY = json.load(open(RESULTS / "results_syniot6.json"))
NS = json.load(open(RESULTS / "results_nslkdd.json"))
R = SY["results"]
DOMS = list(SY["params"].keys())


def ms(v, nd=3): return f"{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}"


def g(mode, key="f1", dom=None):
    rows = [r for r in R if dom is None or r["held"] == dom]
    if mode == "within": return [r["within_f1"] for r in rows]
    if key == "gap": return [r["within_f1"] - r[mode]["f1"] for r in rows]
    return [r[mode][key] for r in rows]


def wt(a, b, key="f1"):
    x, y = np.array(g(a, key)), np.array(g(b, key)); s = wilcoxon(x, y)
    return dict(diff=float((x - y).mean()), wins=int((x > y).sum()), W=float(s.statistic), p=float(s.pvalue))


if __name__ == "__main__":
    print("== SynIoT-6")
    for d in DOMS + [None]:
        print(d, "within", ms(g("within", dom=d)), " ".join(f"{m} {ms(g(m, dom=d))}" for m in ["RAW", "BASE", "FIXED", "XFCA"]),
              "| gap", " ".join(f"{m} {ms(g(m, 'gap', d))}" for m in ["BASE", "FIXED", "XFCA"]))
    for a, b in [("XFCA", "BASE"), ("FIXED", "BASE"), ("XFCA", "FIXED"), ("BASE", "RAW"), ("XFCA", "RAW")]:
        print(a, b, wt(a, b))
    for m in ["BASE", "FIXED", "XFCA"]:
        print(m, "rho_final", ms(g(m, "rho_final")), "probe", ms(g(m, "dom_probe_acc")), "size", round(np.mean(g(m, "size_kb")), 1))
    print("rho XFCA-BASE", wt("XFCA", "BASE", "rho_final"), "probe", wt("XFCA", "BASE", "dom_probe_acc"))
    lg = [l for r in R for l in r["XFCA"]["log"]]
    print("ckpt rho XFCA", ms([l["rho"] for l in lg]), "below tau", sum(l["rho"] < 0.6 for l in lg), "/", len(lg),
          "incr", sum(l["lam_after"] > l["lam_before"] for l in lg), "final lam", ms([r["XFCA"]["log"][-1]["lam_after"] for r in R]))
    print("ckpt rho BASE", ms([l["rho"] for r in R for l in r["BASE"]["log"]]))
    for k in range(3):
        print(" epoch", R[0]["XFCA"]["log"][k]["epoch"], "rho XFCA", ms([r["XFCA"]["log"][k]["rho"] for r in R]),
              "lam", ms([r["XFCA"]["log"][k]["lam_after"] for r in R]))
    a, b = [], []
    for r in R:
        x = np.array([r[m]["rho_final"] for m in ["BASE", "FIXED", "XFCA"]]); y = np.array([r["within_f1"] - r[m]["f1"] for m in ["BASE", "FIXED", "XFCA"]])
        a += list(x - x.mean()); b += list(y - y.mean())
    print("within-fold rho-gap", spearmanr(a, b))
    print("domains won XFCA vs BASE", sum(np.mean(g("XFCA", dom=d)) > np.mean(g("BASE", dom=d)) for d in DOMS))
    print("== NSL-KDD", NS["counts"])
    NR = NS["results"]
    for m in ["RAW", "BASE", "DA"]:
        print(m, "macroF1", ms([r[m]["macro_f1"] for r in NR]), "acc", ms([r[m]["acc"] for r in NR]),
              "pcf1", np.round(np.mean([r[m]["pcf1"] for r in NR], 0), 3).tolist(),
              "pcsd", np.round(np.std([r[m]["pcf1"] for r in NR], 0, ddof=1), 3).tolist(),
              "size", round(np.mean([r[m]["size_kb"] for r in NR]), 1),
              ("unseen" + ms([r[m]["unseen_recall"] for r in NR])) if m != "RAW" else "")
    for key in ["macro_f1", "acc"]:
        x = np.array([r["DA"][key] for r in NR]); y = np.array([r["BASE"][key] for r in NR])
        print(key, "DA-BASE", np.round(x - y, 4).tolist(), "wins", int((x > y).sum()), wilcoxon(x, y).pvalue)
    print("raw subsets", [r["raw_subset"] for r in NR])
