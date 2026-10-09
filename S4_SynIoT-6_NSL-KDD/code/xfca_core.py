"""Generic implementation of XFCA (Algorithm 1) used for the SynIoT-6 and NSL-KDD re-runs.

Identical in every numerical detail to xfca_real.py (Supplementary File S3), but with the
feature and class dimensions taken from the data and with an optional class-loss mask
(used for the transductive NSL-KDD setting, where unlabelled target records enter the
domain branch only).
"""
import numpy as np
import shap
import xgboost as xgb
from scipy.stats import spearmanr

CFG = dict(EPOCHS=90, KC=30, LR=0.01, LAMBDA0=0.3, TAU=0.6, DELTA=0.1, LMIN=0.0, LMAX=1.0,
           M=8, SHAP_N=2000, H1=32, H2=16)
XGB_FINAL = dict(n_estimators=150, max_depth=6, learning_rate=0.15, tree_method="hist", n_jobs=1, verbosity=0)
XGB_SNAP = dict(n_estimators=40, max_depth=4, learning_rate=0.15, tree_method="hist", n_jobs=1, verbosity=0)


def relu(a): return np.maximum(a, 0)


def softmax(a):
    a = a - a.max(1, keepdims=True); e = np.exp(a); return e / e.sum(1, keepdims=True)


def init_params(rng, D, C, K):
    he = lambda i, o: rng.normal(0, np.sqrt(2 / i), (i, o))
    return dict(W1=he(D, CFG["H1"]), b1=np.zeros(CFG["H1"]), W2=he(CFG["H1"], CFG["H2"]), b2=np.zeros(CFG["H2"]),
                Wc=he(CFG["H2"], C), bc=np.zeros(C), Wd1=he(CFG["H2"], 16), bd1=np.zeros(16),
                Wd2=he(16, K), bd2=np.zeros(K))


def encode(P, X):
    return relu(relu(X @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"])


class Adam:
    def __init__(self, P, lr):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in P.items()}; self.v = {k: np.zeros_like(v) for k, v in P.items()}

    def step(self, P, G):
        self.t += 1
        for k in G:
            self.m[k] = 0.9 * self.m[k] + 0.1 * G[k]; self.v[k] = 0.999 * self.v[k] + 0.001 * G[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t); vh = self.v[k] / (1 - 0.999 ** self.t)
            P[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def grad_step(P, X, Y1h, D1h, lam, cmask=None):
    """Full-batch step; the encoder receives dLcls/dz - lam * dLdom/dz (GRL, Eq. 5).
    cmask (bool) restricts the classification loss to labelled records."""
    if cmask is None:
        cmask = np.ones(len(X), bool)
    Nc, Nd = cmask.sum(), len(X)
    a1 = X @ P["W1"] + P["b1"]; h1 = relu(a1); a2 = h1 @ P["W2"] + P["b2"]; z = relu(a2)
    G = {}
    p = softmax(z[cmask] @ P["Wc"] + P["bc"])
    gc = (p - Y1h[cmask]) / Nc
    G["Wc"], G["bc"] = z[cmask].T @ gc, gc.sum(0)
    dz_c = np.zeros_like(z); dz_c[cmask] = gc @ P["Wc"].T
    ad = z @ P["Wd1"] + P["bd1"]; hd = relu(ad); q = softmax(hd @ P["Wd2"] + P["bd2"])
    gd = (q - D1h) / Nd
    G["Wd2"], G["bd2"] = hd.T @ gd, gd.sum(0)
    ghd = (gd @ P["Wd2"].T) * (ad > 0)
    G["Wd1"], G["bd1"] = z.T @ ghd, ghd.sum(0)
    dz_d = ghd @ P["Wd1"].T
    dz = (dz_c - lam * dz_d) * (a2 > 0)
    G["W2"], G["b2"] = h1.T @ dz, dz.sum(0)
    dh1 = (dz @ P["W2"].T) * (a1 > 0)
    G["W1"], G["b1"] = X.T @ dh1, dh1.sum(0)
    Lc = -np.mean(np.log(p[Y1h[cmask].astype(bool)] + 1e-12))
    Ld = -np.mean(np.log(q[D1h.astype(bool)] + 1e-12))
    return G, Lc, Ld


def mean_abs_shap(model, U, offset):
    sv = np.asarray(shap.TreeExplainer(model).shap_values(U))
    if sv.ndim == 3 and sv.shape[0] == len(U):
        imp = np.abs(sv).mean(axis=(0, 2))
    elif sv.ndim == 3:
        imp = np.abs(sv).mean(axis=(0, 1))
    else:
        imp = np.abs(sv).mean(0)
    return imp[offset:]


def spearman_topm(a, b, m):
    ta, tb = np.argsort(-a)[:m], np.argsort(-b)[:m]
    union = sorted(set(ta) | set(tb))
    ra = {f: i + 1 for i, f in enumerate(ta)}; rb = {f: i + 1 for i, f in enumerate(tb)}
    return float(spearmanr([ra.get(f, m + 1) for f in union], [rb.get(f, m + 1) for f in union]).correlation)


def consistency(model, U_by_dom, offset, m=None):
    m = m or CFG["M"]
    imps = [mean_abs_shap(model, U, offset) for U in U_by_dom]
    pairs = [spearman_topm(imps[a], imps[b], m) for a in range(len(imps)) for b in range(a + 1, len(imps))]
    return float(np.mean(pairs)), imps


def train_hybrid(mode, Xs, y, dom, C, K, seed, shap_idx=None, cmask=None, raw_idx=None):
    """mode: BASE (lambda=0), FIXED (lambda=lambda0), XFCA (Eq. 10), RAMP (SHAP-blind), DA (fixed lambda0, no checkpoints).
    raw_idx: columns of Xs passed to the decision layer as x_S (default: all)."""
    rng = np.random.default_rng(seed)
    D = Xs.shape[1]
    P = init_params(rng, D, C, K); opt = Adam(P, CFG["LR"])
    Y1h = np.zeros((len(Xs), C)); lab = cmask if cmask is not None else np.ones(len(Xs), bool)
    Y1h[np.where(lab)[0], y[lab]] = 1
    D1h = np.eye(K)[dom]
    lam = 0.0 if mode == "BASE" else CFG["LAMBDA0"]
    raw_idx = np.arange(D) if raw_idx is None else np.asarray(raw_idx)
    log = []
    for e in range(1, CFG["EPOCHS"] + 1):
        G, Lc, Ld = grad_step(P, Xs, Y1h, D1h, lam, cmask)
        opt.step(P, G)
        if e % CFG["KC"] == 0 and mode in ("BASE", "FIXED", "XFCA") and shap_idx is not None:
            U = np.hstack([encode(P, Xs[lab]), Xs[lab][:, raw_idx]])
            snap = xgb.XGBClassifier(**XGB_SNAP, random_state=seed).fit(U, y[lab])
            Uall = np.hstack([encode(P, Xs), Xs[:, raw_idx]])
            rho, _ = consistency(snap, [Uall[i] for i in shap_idx], CFG["H2"])
            lb = lam
            if mode == "XFCA":
                lam = min(lam + CFG["DELTA"], CFG["LMAX"]) if rho < CFG["TAU"] else max(lam - CFG["DELTA"] / 3, CFG["LMIN"])
            log.append(dict(epoch=e, rho=rho, lam_before=lb, lam_after=lam, Lcls=float(Lc), Ldom=float(Ld)))
        elif e % CFG["KC"] == 0 and mode == "RAMP":
            lb = lam; lam = min(lam + CFG["DELTA"], CFG["LMAX"])
            log.append(dict(epoch=e, rho=float("nan"), lam_before=lb, lam_after=lam, Lcls=float(Lc), Ldom=float(Ld)))
    U = np.hstack([encode(P, Xs[lab]), Xs[lab][:, raw_idx]])
    final = xgb.XGBClassifier(**XGB_FINAL, random_state=seed).fit(U, y[lab])
    return P, final, log


def decision_input(P, X, raw_idx=None):
    raw_idx = np.arange(X.shape[1]) if raw_idx is None else np.asarray(raw_idx)
    return np.hstack([encode(P, X), X[:, raw_idx]])


def size_kb(model):
    return len(model.get_booster().save_raw("json")) / 1024
