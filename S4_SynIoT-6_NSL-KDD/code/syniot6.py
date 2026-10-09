"""SynIoT-6: six-domain synthetic IoT intrusion benchmark (generator) and the leave-one-domain-out
re-run of XFCA (Algorithm 1). Deterministic: the benchmark is fixed by GEN_SEED; experiment seeds
control bootstrap resampling, splits and initialization only.
"""
import json, sys, time
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("XFCA_RESULTS", ROOT / "results"))
DATA = os.environ.get("XFCA_DATA", str(ROOT / "data"))
import numpy as np
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import xfca_core as xc

GEN_SEED = 2024
N_PER_DOMAIN = 1400
CLASSES = ["Benign", "DoS/DDoS", "Reconnaissance", "BruteForce", "Botnet", "Web/Injection"]
FEATURES = ["duration", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "iat_mean", "iat_std",
            "pkt_size_mean", "pkt_size_std", "syn_flag_cnt", "ack_flag_cnt", "rst_flag_cnt",
            "fin_flag_cnt", "dst_port_diversity", "proto_tcp", "proto_udp"]
SIG_SD = 1.5                      # spread of the shared class signatures
DOMAINS = {  # name: (class priors in CLASSES order, noise sd, scale range, shift sd)
    "D1 (CICIoT2023-like)":   ([0.10, 0.55, 0.10, 0.05, 0.15, 0.05], 0.8, (0.7, 1.3), 0.6),
    "D2 (TON_IoT-like)":      ([0.35, 0.20, 0.15, 0.10, 0.05, 0.15], 0.9, (0.7, 1.3), 0.8),
    "D3 (N-BaIoT-like)":      ([0.15, 0.30, 0.10, 0.05, 0.38, 0.02], 1.0, (0.6, 1.4), 0.8),
    "D4 (BoT-IoT-like)":      ([0.02, 0.70, 0.20, 0.02, 0.03, 0.03], 1.2, (0.5, 1.5), 1.2),
    "D5 (IoT-23-like)":       ([0.25, 0.15, 0.25, 0.05, 0.27, 0.03], 1.1, (0.6, 1.4), 1.0),
    "D6 (Edge-IIoTset-like)": ([0.40, 0.20, 0.12, 0.08, 0.05, 0.15], 1.0, (0.6, 1.4), 1.0),
}


def generate():
    rng = np.random.default_rng(GEN_SEED)
    sig = rng.normal(0, SIG_SD, (len(CLASSES), len(FEATURES)))        # shared latent attack signatures
    data, params = {}, {}
    for name, (pri, noise, (lo, hi), ssd) in DOMAINS.items():
        a = rng.uniform(lo, hi, len(FEATURES)); b = rng.normal(0, ssd, len(FEATURES))
        counts = np.round(np.array(pri) * N_PER_DOMAIN).astype(int)
        counts[np.argmax(counts)] += N_PER_DOMAIN - counts.sum()
        y = np.repeat(np.arange(len(CLASSES)), counts)
        X = sig[y] + rng.normal(0, noise, (len(y), len(FEATURES)))
        X = a * X + b                                                    # domain-specific affine nuisance
        perm = rng.permutation(len(y))
        data[name] = (X[perm], y[perm])
        params[name] = dict(priors=pri, counts=counts.tolist(), noise=noise, scale_range=[lo, hi], shift_sd=ssd,
                            scale_drawn=[float(a.min()), float(a.max())], shift_drawn=[float(b.min()), float(b.max())])
    return data, params


def mf1(y, p): return float(f1_score(y, p, average="macro", labels=list(range(len(CLASSES))), zero_division=0))


def run(seeds=(42, 43, 44), modes=("BASE", "FIXED", "XFCA")):
    data, params = generate()
    names = list(DOMAINS)
    C = len(CLASSES)
    res, t0 = [], time.time()
    for seed in seeds:
        for held in names:
            srcs = [k for k in names if k != held]
            rng = np.random.default_rng(seed)
            Xtr, ytr, dtr = [], [], []
            for i, k in enumerate(srcs):                  # bootstrap resampling of each source domain
                X, y = data[k]; idx = rng.integers(0, len(y), len(y))
                Xtr.append(X[idx]); ytr.append(y[idx]); dtr.append(np.full(len(y), i))
            Xtr, ytr, dtr = np.vstack(Xtr), np.concatenate(ytr), np.concatenate(dtr)
            Xh, yh = data[held]
            hi_tr, hi_te = train_test_split(np.arange(len(yh)), test_size=0.3, stratify=yh, random_state=seed)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8          # standardization: pooled source domains
            Xs, Xte = (Xtr - mu) / sd, (Xh[hi_te] - mu) / sd
            yte = yh[hi_te]
            r = np.random.default_rng(seed + 1000)
            shap_idx = [r.choice(np.where(dtr == i)[0], min(xc.CFG["SHAP_N"], (dtr == i).sum()), replace=False)
                        for i in range(len(srcs))]
            rec = dict(seed=seed, held=held)
            # within-domain reference: baseline hybrid on 70% of the held-out domain
            Xw = Xh[hi_tr]; mw, sw = Xw.mean(0), Xw.std(0) + 1e-8
            Pw, fw, _ = xc.train_hybrid("BASE", (Xw - mw) / sw, yh[hi_tr], np.zeros(len(hi_tr), int), C, 1, seed)
            rec["within_f1"] = mf1(yte, fw.predict(xc.decision_input(Pw, (Xh[hi_te] - mw) / sw)))
            raw = xgb.XGBClassifier(**xc.XGB_FINAL, random_state=seed).fit(Xs, ytr)
            rec["RAW"] = dict(f1=mf1(yte, raw.predict(Xte)), size_kb=xc.size_kb(raw))
            pr_tr, pr_te = train_test_split(np.arange(len(dtr)), test_size=0.3, stratify=dtr, random_state=seed)
            for mode in modes:
                P, final, log = xc.train_hybrid(mode, Xs, ytr, dtr, C, len(srcs), seed, shap_idx)
                Utr = xc.decision_input(P, Xs)
                pred = final.predict(xc.decision_input(P, Xte))
                rho_f, _ = xc.consistency(final, [Utr[i] for i in shap_idx], xc.CFG["H2"])
                Z = xc.encode(P, Xs)
                probe = LogisticRegression(max_iter=3000).fit(Z[pr_tr], dtr[pr_tr])
                rec[mode] = dict(f1=mf1(yte, pred), rho_final=rho_f, size_kb=xc.size_kb(final),
                                 dom_probe_acc=float(probe.score(Z[pr_te], dtr[pr_te])), log=log)
            res.append(rec)
            print(f"seed {seed} {held:24s} within {rec['within_f1']:.3f} RAW {rec['RAW']['f1']:.3f} " +
                  " ".join(f"{m} {rec[m]['f1']:.3f}" for m in modes) +
                  f" | lam {[round(l['lam_after'], 2) for l in rec['XFCA']['log']]} rho {[round(l['rho'], 2) for l in rec['XFCA']['log']]} ({time.time()-t0:.0f}s)",
                  flush=True)
    return res, params


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":   # benchmark sanity check (raw XGBoost only)
        data, params = generate()
        for k, (X, y) in data.items():
            a, b, ya, yb = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
            m = xgb.XGBClassifier(**xc.XGB_FINAL).fit(a, ya)
            print(k, params[k]["counts"], "within-domain raw-XGB macro-F1", round(mf1(yb, m.predict(b)), 3))
        sys.exit()
    res, params = run()
    json.dump(dict(cfg=xc.CFG, gen_seed=GEN_SEED, params=params, results=res),
              open(RESULTS / "results_syniot6.json", "w"), indent=1)
    print("saved")
