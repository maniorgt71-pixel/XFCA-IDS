"""Leave-one-domain-out evaluation of the complete XFCA training algorithm
(Algorithm 1: FFNN encoder + gradient-reversal domain branch + SHAP-consistency
control of lambda + XGBoost decision layer) on three real, independently
collected datasets harmonized to one schema (harmonize.py).

Conditions (paired: same subsample, split and initialization per fold/seed):
  RAW    raw XGBoost on the harmonized features (pooled-source training)
  BASE   hybrid, lambda fixed at 0, no checkpoint updates (Section 4.2.1 baseline)
  FIXED  hybrid, lambda fixed at lambda_0 = 0.3, no SHAP update (isolates SHAP rule)
  XFCA   hybrid, lambda adapted by Eq. (10) from TreeSHAP consistency (Eq. 8-9)
"""
import json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import shap, xgboost as xgb
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from harmonize import load_all, DOMAINS, CLASSES, FEATURES

warnings.filterwarnings("ignore")

CFG = dict(N_PER_DOMAIN=15000, SEEDS=[42, 43, 44, 45, 46], EPOCHS=90, KC=30,
           LR=0.01, LAMBDA0=0.3, TAU=0.6, DELTA=0.1, LMIN=0.0, LMAX=1.0, M=8,
           SHAP_N=2000, H1=32, H2=16)
if len(sys.argv) > 1:  # optional overrides: epochs, tau
    CFG["EPOCHS"] = int(sys.argv[1])
if len(sys.argv) > 2:
    CFG["TAU"] = float(sys.argv[2])
import os
MODES = os.environ.get("MODES", "BASE,FIXED,XFCA").split(",")
ONLY = os.environ.get("ONLY_MODES") == "1"   # skip within/RAW when re-running a subset
XGB_FINAL = dict(n_estimators=150, max_depth=6, learning_rate=0.15,
                 tree_method="hist", n_jobs=1, verbosity=0)
XGB_SNAP = dict(n_estimators=40, max_depth=4, learning_rate=0.15,
                tree_method="hist", n_jobs=1, verbosity=0)
C, D = len(CLASSES), len(FEATURES)


# ---------------------------------------------------------------- FFNN (NumPy)
def relu(a): return np.maximum(a, 0)


def softmax(a):
    a = a - a.max(1, keepdims=True); e = np.exp(a); return e / e.sum(1, keepdims=True)


def init_params(rng, K):
    he = lambda i, o: rng.normal(0, np.sqrt(2 / i), (i, o))
    return dict(W1=he(D, CFG["H1"]), b1=np.zeros(CFG["H1"]),
                W2=he(CFG["H1"], CFG["H2"]), b2=np.zeros(CFG["H2"]),
                Wc=he(CFG["H2"], C), bc=np.zeros(C),
                Wd1=he(CFG["H2"], 16), bd1=np.zeros(16),
                Wd2=he(16, K), bd2=np.zeros(K))


def encode(P, X):
    return relu(relu(X @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"])


class Adam:
    def __init__(self, P, lr):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}

    def step(self, P, G):
        self.t += 1
        for k in G:
            self.m[k] = 0.9 * self.m[k] + 0.1 * G[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * G[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t); vh = self.v[k] / (1 - 0.999 ** self.t)
            P[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def grad_step(P, X, Y1h, D1h, lam):
    """One full-batch step. Encoder receives dL_cls/dz - lam * dL_dom/dz (GRL, Eq. 5)."""
    N = len(X)
    a1 = X @ P["W1"] + P["b1"]; h1 = relu(a1)
    a2 = h1 @ P["W2"] + P["b2"]; z = relu(a2)
    p = softmax(z @ P["Wc"] + P["bc"])
    ad = z @ P["Wd1"] + P["bd1"]; hd = relu(ad)
    q = softmax(hd @ P["Wd2"] + P["bd2"])
    Lc = -np.mean(np.log(p[Y1h.astype(bool)] + 1e-12))
    Ld = -np.mean(np.log(q[D1h.astype(bool)] + 1e-12))
    G = {}
    gc = (p - Y1h) / N
    G["Wc"], G["bc"] = z.T @ gc, gc.sum(0)
    dz_c = gc @ P["Wc"].T
    gd = (q - D1h) / N
    G["Wd2"], G["bd2"] = hd.T @ gd, gd.sum(0)
    ghd = (gd @ P["Wd2"].T) * (ad > 0)
    G["Wd1"], G["bd1"] = z.T @ ghd, ghd.sum(0)
    dz_d = ghd @ P["Wd1"].T
    dz = (dz_c - lam * dz_d) * (a2 > 0)          # gradient reversal
    G["W2"], G["b2"] = h1.T @ dz, dz.sum(0)
    dh1 = (dz @ P["W2"].T) * (a1 > 0)
    G["W1"], G["b1"] = X.T @ dh1, dh1.sum(0)
    return G, Lc, Ld


# ------------------------------------------------------- SHAP consistency (Eq. 8-9)
def topm_ranking(model, U, n_raw_offset):
    """Mean |TreeSHAP| of raw features (embedding dims excluded), averaged over classes."""
    sv = shap.TreeExplainer(model).shap_values(U)
    sv = np.asarray(sv)
    if sv.ndim == 3 and sv.shape[0] == len(U):      # (n, f, classes)
        imp = np.abs(sv).mean(axis=(0, 2))
    elif sv.ndim == 3:                               # (classes, n, f)
        imp = np.abs(sv).mean(axis=(0, 1))
    else:
        imp = np.abs(sv).mean(0)
    imp = imp[n_raw_offset:]
    return imp


def spearman_topm(imp_a, imp_b, m):
    """Spearman correlation of two top-m rankings over the union of their features;
    a feature absent from one list receives rank m+1 in that list."""
    ta, tb = np.argsort(-imp_a)[:m], np.argsort(-imp_b)[:m]
    union = sorted(set(ta) | set(tb))
    ra = {f: i + 1 for i, f in enumerate(ta)}; rb = {f: i + 1 for i, f in enumerate(tb)}
    x = [ra.get(f, m + 1) for f in union]; y = [rb.get(f, m + 1) for f in union]
    return float(spearmanr(x, y).correlation)


def consistency(model, U_by_dom, offset, m):
    imps = [topm_ranking(model, U, offset) for U in U_by_dom]
    pairs = [spearman_topm(imps[a], imps[b], m)
             for a in range(len(imps)) for b in range(a + 1, len(imps))]
    return float(np.mean(pairs)), imps


# ------------------------------------------------------------------ training
def train_hybrid(mode, Xs, y, dsrc, K, seed, shap_idx):
    rng = np.random.default_rng(seed)
    P = init_params(rng, K)
    opt = Adam(P, CFG["LR"])
    Y1h = np.eye(C)[y]; D1h = np.eye(K)[dsrc]
    lam = 0.0 if mode == "BASE" else CFG["LAMBDA0"]
    log = []
    for e in range(1, CFG["EPOCHS"] + 1):
        G, Lc, Ld = grad_step(P, Xs, Y1h, D1h, lam)
        opt.step(P, G)
        if e % CFG["KC"] == 0 and mode == "RAMP":
            # SHAP-blind control: same step rule as XFCA, but lambda always increases
            lam_before = lam; lam = min(lam + CFG["DELTA"], CFG["LMAX"])
            log.append(dict(epoch=e, rho=float("nan"), lam_before=lam_before, lam_after=lam,
                            Lcls=float(Lc), Ldom=float(Ld)))
        elif e % CFG["KC"] == 0:
            Z = encode(P, Xs); U = np.hstack([Z, Xs])
            snap = xgb.XGBClassifier(**XGB_SNAP, random_state=seed).fit(U, y)
            rho, _ = consistency(snap, [U[i] for i in shap_idx], CFG["H2"], CFG["M"])
            lam_before = lam
            if mode == "XFCA":
                lam = min(lam + CFG["DELTA"], CFG["LMAX"]) if rho < CFG["TAU"] \
                    else max(lam - CFG["DELTA"] / 3, CFG["LMIN"])
            log.append(dict(epoch=e, rho=rho, lam_before=lam_before, lam_after=lam,
                            Lcls=float(Lc), Ldom=float(Ld)))
    Z = encode(P, Xs); U = np.hstack([Z, Xs])
    final = xgb.XGBClassifier(**XGB_FINAL, random_state=seed).fit(U, y)
    return P, final, log


def macro_f1(y, p): return float(f1_score(y, p, average="macro", labels=list(range(C)), zero_division=0))


def per_class_f1(y, p): return f1_score(y, p, average=None, labels=list(range(C)), zero_division=0).tolist()


def run():
    data = load_all()
    results, t0 = [], time.time()
    for seed in CFG["SEEDS"]:
        rng = np.random.default_rng(seed)
        # stratified per-domain subsample of N records (native class priors kept)
        sub = {}
        for k in DOMAINS:
            X, y = data[k][0], data[k][1]
            idx, _ = train_test_split(np.arange(len(y)), train_size=CFG["N_PER_DOMAIN"],
                                      stratify=y, random_state=seed)
            sub[k] = (X[idx], y[idx])
        for held in DOMAINS:
            srcs = [k for k in DOMAINS if k != held]
            Xtr = np.vstack([sub[k][0] for k in srcs]); ytr = np.concatenate([sub[k][1] for k in srcs])
            dtr = np.concatenate([np.full(len(sub[k][1]), i) for i, k in enumerate(srcs)])
            Xh, yh = sub[held]
            # 70/30 split of the held-out domain: within-domain reference trains on 70%,
            # every model is scored on the same 30%
            hi_tr, hi_te = train_test_split(np.arange(len(yh)), test_size=0.3,
                                            stratify=yh, random_state=seed)
            # standardization fitted on the pooled source domains only
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
            Xtr_s, Xte_s = (Xtr - mu) / sd, (Xh[hi_te] - mu) / sd
            yte = yh[hi_te]
            r = np.random.default_rng(seed + 1000)
            shap_idx = [r.choice(np.where(dtr == i)[0], CFG["SHAP_N"], replace=False)
                        for i in range(len(srcs))]
            # dom-probe split (source records only)
            pr_tr, pr_te = train_test_split(np.arange(len(dtr)), test_size=0.3,
                                            stratify=dtr, random_state=seed)
            rec = dict(seed=seed, held=held, sources=srcs, n_train=len(ytr), n_test=len(yte))

            if ONLY:
                for mode in MODES:
                    P, final, log = train_hybrid(mode, Xtr_s, ytr, dtr, len(srcs), seed, shap_idx)
                    Ztr = encode(P, Xtr_s); Utr = np.hstack([Ztr, Xtr_s])
                    pred = final.predict(np.hstack([encode(P, Xte_s), Xte_s]))
                    rho_f, _ = consistency(final, [Utr[i] for i in shap_idx], CFG["H2"], CFG["M"])
                    probe = LogisticRegression(max_iter=2000).fit(Ztr[pr_tr], dtr[pr_tr])
                    rec[mode] = dict(f1=macro_f1(yte, pred), acc=float(accuracy_score(yte, pred)),
                                     pcf1=per_class_f1(yte, pred), rho_final=rho_f,
                                     dom_probe_acc=float(probe.score(Ztr[pr_te], dtr[pr_te])), log=log)
                results.append(rec); print(seed, held, {m: round(rec[m]["f1"], 3) for m in MODES}, flush=True)
                continue
            # within-domain reference (baseline hybrid on the held-out domain's 70%)
            Xw = Xh[hi_tr]; mw, sw = Xw.mean(0), Xw.std(0) + 1e-8
            # (K=1: no domain branch; the hybrid is trained with lambda = 0)
            rngw = np.random.default_rng(seed); Pw = init_params(rngw, 1); optw = Adam(Pw, CFG["LR"])
            Xws = (Xw - mw) / sw; Y1 = np.eye(C)[yh[hi_tr]]; D1 = np.ones((len(Xws), 1))
            for _ in range(CFG["EPOCHS"]):
                G, _, _ = grad_step(Pw, Xws, Y1, D1, 0.0); optw.step(Pw, G)
            Uw = np.hstack([encode(Pw, Xws), Xws])
            fw = xgb.XGBClassifier(**XGB_FINAL, random_state=seed).fit(Uw, yh[hi_tr])
            Xte_w = (Xh[hi_te] - mw) / sw
            pw = fw.predict(np.hstack([encode(Pw, Xte_w), Xte_w]))
            rec["within_f1"] = macro_f1(yte, pw)

            # RAW XGBoost on harmonized features
            raw = xgb.XGBClassifier(**XGB_FINAL, random_state=seed).fit(Xtr_s, ytr)
            pr = raw.predict(Xte_s)
            rho_raw, _ = consistency(raw, [Xtr_s[i] for i in shap_idx], 0, CFG["M"])
            rec["RAW"] = dict(f1=macro_f1(yte, pr), acc=float(accuracy_score(yte, pr)),
                              pcf1=per_class_f1(yte, pr), rho_final=rho_raw)

            for mode in MODES:
                P, final, log = train_hybrid(mode, Xtr_s, ytr, dtr, len(srcs), seed, shap_idx)
                Ztr = encode(P, Xtr_s); Utr = np.hstack([Ztr, Xtr_s])
                Ute = np.hstack([encode(P, Xte_s), Xte_s])
                pred = final.predict(Ute)
                rho_f, imps = consistency(final, [Utr[i] for i in shap_idx], CFG["H2"], CFG["M"])
                probe = LogisticRegression(max_iter=2000).fit(Ztr[pr_tr], dtr[pr_tr])
                rec[mode] = dict(f1=macro_f1(yte, pred), acc=float(accuracy_score(yte, pred)),
                                 pcf1=per_class_f1(yte, pred), rho_final=rho_f,
                                 dom_probe_acc=float(probe.score(Ztr[pr_te], dtr[pr_te])),
                                 src_f1=macro_f1(ytr, final.predict(Utr)),
                                 size_kb=len(final.get_booster().save_raw("json")) / 1024,
                                 log=log, top_feats=[[FEATURES[j] for j in np.argsort(-im)[:5]] for im in imps])
            results.append(rec)
            print(f"seed {seed} held {held:11s} within {rec['within_f1']:.3f} | " +
                  " ".join(f"{m} {rec[m]['f1']:.3f}" for m in ["RAW", "BASE", "FIXED", "XFCA"]) +
                  f" | lam XFCA {[round(l['lam_after'],2) for l in rec['XFCA']['log']]}"
                  f" rho {[round(l['rho'],2) for l in rec['XFCA']['log']]}  ({time.time()-t0:.0f}s)",
                  flush=True)
    return results


if __name__ == "__main__":
    res = run()
    out = Path(os.environ.get("XFCA_RESULTS", Path(__file__).resolve().parent.parent / "results")) / f"results_E{CFG['EPOCHS']}_tau{CFG['TAU']}{'_'+'-'.join(MODES) if ONLY else ''}.json"
    json.dump(dict(cfg=CFG, results=res), open(out, "w"), indent=1)
    print("saved", out)
