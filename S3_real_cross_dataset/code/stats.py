import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("XFCA_RESULTS", ROOT / "results"))
import numpy as np
from scipy.stats import wilcoxon, spearmanr

DOM = ["NSL-KDD", "UNSW-NB15", "RT-IoT2022"]
M = json.load(open(RESULTS / "results_E90_tau0.6.json"))
S = json.load(open(RESULTS / "results_E300_tau0.95.json"))
RA = json.load(open(RESULTS / "results_E300_tau0.95_RAMP.json"))["results"]
for r, q in zip(S["results"], RA):
    assert r["seed"] == q["seed"] and r["held"] == q["held"]
    r["RAMP"] = q["RAMP"]
M, S = M["results"], S["results"]


EPS = 1e-9  # lambda changes smaller than this are floating-point rounding, not updates


def f(v, sd=True, nd=3):
    return f"{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}" if sd else f"{np.mean(v):.{nd}f}"


def get(R, mode, key="f1", dom=None):
    rows = [r for r in R if dom is None or r["held"] == dom]
    if mode == "within":
        return [r["within_f1"] for r in rows]
    if key == "gap":
        return [r["within_f1"] - r[mode]["f1"] for r in rows]
    return [r[mode][key] for r in rows]


def test(R, a, b, key="f1"):
    x = np.array(get(R, a, key)); y = np.array(get(R, b, key))
    st = wilcoxon(x, y)
    return dict(diff=float(np.mean(x - y)), wins=int((x > y).sum()), ties=int((x == y).sum()),
                W=float(st.statistic), p=float(st.pvalue))


if __name__ == "__main__":
    for name, R, modes in [("MAIN", M, ["RAW", "BASE", "FIXED", "XFCA"]),
                           ("SENS", S, ["RAW", "BASE", "FIXED", "RAMP", "XFCA"])]:
        print("=====", name)
        for d in DOM + [None]:
            print(d, "within", f(get(R, "within", dom=d)), " | ".join(f"{m} {f(get(R, m, dom=d))}" for m in modes))
        print("acc", " | ".join(f"{m} {f(get(R, m, 'acc'))}" for m in modes))
        print("gap", " | ".join(f"{m} {f(get(R, m, 'gap'))}" for m in modes))
        for a, b in [("XFCA", "BASE"), ("FIXED", "BASE"), ("XFCA", "FIXED"), ("XFCA", "RAW"), ("BASE", "RAW")] + (
                [("RAMP", "BASE"), ("XFCA", "RAMP")] if name == "SENS" else []):
            print(a, "vs", b, test(R, a, b))
        for m in modes:
            line = f"{m} rho_final {f(get(R, m, 'rho_final'))}"
            if m != "RAW":
                line += f" probe {f(get(R, m, 'dom_probe_acc'))}"
            print(line)
        print("probe XFCA vs BASE", test(R, "XFCA", "BASE", "dom_probe_acc"))
        print("rho XFCA vs BASE", test(R, "XFCA", "BASE", "rho_final"))
        lam = [r["XFCA"]["log"][-1]["lam_after"] for r in R]
        inc = sum(1 for r in R for l in r["XFCA"]["log"] if l["lam_after"] - l["lam_before"] > EPS)
        dec = sum(1 for r in R for l in r["XFCA"]["log"] if l["lam_before"] - l["lam_after"] > EPS)
        n = sum(len(r["XFCA"]["log"]) for r in R)
        eff = sum(1 for r in R for l in r["XFCA"]["log"][:-1] if l["lam_after"] - l["lam_before"] > EPS)
        neff = sum(len(r["XFCA"]["log"]) - 1 for r in R)
        print(f"lambda final {f(lam)}; increases {inc}/{n}, decreases {dec}/{n}; effective increases {eff}/{neff}")
        print("ckpt rho XFCA", f([l["rho"] for r in R for l in r["XFCA"]["log"]]),
              "BASE", f([l["rho"] for r in R for l in r["BASE"]["log"]]))
        for d in DOM:
            for m in modes:
                pc = np.array([r[m]["pcf1"] for r in R if r["held"] == d]).mean(0)
                print("  pcF1", d, m, np.round(pc, 3).tolist())
        hyb = [m for m in modes if m not in ("RAW",)]
        rr = [r[m]["rho_final"] for r in R for m in hyb]; gg = [r["within_f1"] - r[m]["f1"] for r in R for m in hyb]
        a = [];
        b = []
        for r in R:
            x = np.array([r[m]["rho_final"] for m in hyb]); y = np.array([r["within_f1"] - r[m]["f1"] for m in hyb])
            a += list(x - x.mean()); b += list(y - y.mean())
        print("rho-gap pooled", spearmanr(rr, gg), "within-fold", spearmanr(a, b))
        if name == "SENS":
            x = np.array(get(R, "XFCA")); y = np.array(get(R, "RAMP"))
            print("XFCA-RAMP abs diff per fold", np.round(x - y, 3).tolist())
            print("RT per seed XFCA", np.round(get(R, "XFCA", dom="RT-IoT2022"), 3).tolist(),
                  "RAMP", np.round(get(R, "RAMP", dom="RT-IoT2022"), 3).tolist(),
                  "FIXED", np.round(get(R, "FIXED", dom="RT-IoT2022"), 3).tolist(),
                  "BASE", np.round(get(R, "BASE", dom="RT-IoT2022"), 3).tolist())
            print("RAMP increases", sum(1 for r in R for l in r["RAMP"]["log"] if l["lam_after"] - l["lam_before"] > EPS), "/", sum(len(r["RAMP"]["log"]) for r in R))
            print("RAMP probe", test(R, "RAMP", "BASE", "dom_probe_acc"))
