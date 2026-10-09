"""
rt_iot2022.py - five-seed RT-IoT2022 validation of the hybrid FFNN-XGBoost pipeline
(Sections 4.1.4, 4.2.5 and 4.6; Algorithm 3), experiments E1-E4.
Usage:  python rt_iot2022.py            -> results_rtiot/results.json        (manuscript protocol, Tables 13-16)
        python rt_iot2022.py --dedup    -> results_rtiot_dedup/results.json  (duplicate-free protocol, Section 4.6.6)
In --dedup mode, port-free feature vectors that occur with more than one label are removed, and of every
remaining set of identical port-free feature vectors only the first record is kept, so that no test record
has an exact copy in the training split.
"""
import hashlib, json, os, pickle, sys, time, urllib.request
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import xgboost as xgb
import shap

URL = "https://raw.githubusercontent.com/Binwakil/RT_IOT2022_IDS/main/Data/RT_IOT2022.csv"
SHA = "d9b315d6d899cd60d356f7d3dbe61907f1163985bd800a87cffe600eb577bd20"
SEEDS = [42, 43, 44, 45, 46]
DEDUP = "--dedup" in sys.argv
OUT = "results_rtiot_dedup" if DEDUP else "results_rtiot"; os.makedirs(OUT, exist_ok=True)
COARSE = {"Thing_Speak": "Benign", "MQTT_Publish": "Benign", "Wipro_bulb": "Benign",
          "DOS_SYN_Hping": "DoS/DDoS", "DDOS_Slowloris": "DoS/DDoS",
          "NMAP_UDP_SCAN": "Reconnaissance", "NMAP_XMAS_TREE_SCAN": "Reconnaissance",
          "NMAP_OS_DETECTION": "Reconnaissance", "NMAP_TCP_scan": "Reconnaissance",
          "NMAP_FIN_SCAN": "Reconnaissance", "Metasploit_Brute_Force_SSH": "BruteForce",
          "ARP_poisioning": "Spoofing/MITM"}
CLASSES = ["Benign", "DoS/DDoS", "Reconnaissance", "BruteForce", "Spoofing/MITM"]
WITHHELD = ["NMAP_XMAS_TREE_SCAN", "NMAP_FIN_SCAN", "DDOS_Slowloris", "Metasploit_Brute_Force_SSH"]
XP = dict(n_estimators=150, max_depth=6, learning_rate=0.15, n_jobs=1)

if not os.path.exists("RT_IOT2022.csv"):
    urllib.request.urlretrieve(URL, "RT_IOT2022.csv")
assert hashlib.sha256(open("RT_IOT2022.csv", "rb").read()).hexdigest() == SHA, "checksum mismatch"
df = pd.read_csv("RT_IOT2022.csv")
sub = df.Attack_type.to_numpy()
df = df.drop(columns=["no", "id.orig_p", "id.resp_p", "Attack_type"])
df = pd.get_dummies(df, columns=["proto", "service"], dtype=float)
names = list(df.columns); X = df.to_numpy(float)
if DEDUP:
    key = pd.util.hash_pandas_object(df, index=False).to_numpy()
    k = pd.DataFrame({"k": key, "l": sub})
    n_lab = k.groupby("k").l.nunique()
    keep = (~k.k.isin(set(n_lab[n_lab > 1].index)) & ~k.k.duplicated()).to_numpy()
    X, sub = X[keep], sub[keep]
y5 = np.array([CLASSES.index(COARSE[s]) for s in sub]); yb = (y5 != 0).astype(int)
print(f"{'duplicate-free' if DEDUP else 'original'}: {X.shape[0]} records, {X.shape[1]} features, "
      f"class counts {np.bincount(y5, minlength=5).tolist()}")


def run_pair(Xtr, ytr, seed, n_cls):
    B = xgb.XGBClassifier(random_state=seed, **XP).fit(Xtr, ytr)
    S = list(np.argsort(-B.feature_importances_)[:3])
    sc = StandardScaler().fit(Xtr)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=300,
                        random_state=seed).fit(sc.transform(Xtr), ytr)

    def feats(X_):
        Z = sc.transform(X_)
        Z = np.maximum(0, np.maximum(0, Z @ mlp.coefs_[0] + mlp.intercepts_[0]) @ mlp.coefs_[1] + mlp.intercepts_[1])
        return np.hstack([Z, X_[:, S]])
    H = xgb.XGBClassifier(random_state=seed, **XP).fit(feats(Xtr), ytr)
    return B, H, feats, S, sc, mlp


res = {"E1": {"5class": [], "binary": []}, "E3": [], "S": []}
models = {}
# ---- E1
for task, y in [("5class", y5), ("binary", yb)]:
    for seed in SEEDS:
        t0 = time.time()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        B, H, feats, S, sc, mlp = run_pair(Xtr, ytr, seed, len(np.unique(y)))
        pb, ph = B.predict(Xte), H.predict(feats(Xte))
        row = dict(seed=seed, n_test=len(yte), S=[names[i] for i in S])
        for nm, p in [("base", pb), ("hyb", ph)]:
            if task == "5class":
                P, R, F, _ = precision_recall_fscore_support(yte, p, average="macro", zero_division=0)
                row[nm] = dict(acc=accuracy_score(yte, p), macro_p=P, macro_r=R, macro_f1=F,
                               weighted_f1=f1_score(yte, p, average="weighted"),
                               per_class=f1_score(yte, p, average=None, labels=range(5)).tolist())
            else:
                P, R, F, _ = precision_recall_fscore_support(yte, p, average="binary")
                row[nm] = dict(acc=accuracy_score(yte, p), precision=P, recall=R, f1=F)
        res["E1"][task].append(row)
        models[(task, seed)] = (B, H, feats, sc, mlp, Xte)
        print(f"E1 {task} seed {seed}: base acc {row['base']['acc']:.4f} hyb acc {row['hyb']['acc']:.4f} "
              f"S={row['S']} ({time.time() - t0:.0f}s)", flush=True)

# ---- E2: TreeSHAP on the 5-class baseline (seed 42) and stability across binary baselines
rng = np.random.default_rng(42)
B, _, _, _, _, Xte = models[("5class", 42)]
sv = np.abs(np.asarray(shap.TreeExplainer(B).shap_values(Xte[rng.choice(len(Xte), 2000, replace=False)])))
imp = sv.mean(axis=(0, 2)) if sv.ndim == 3 else sv.mean(0)
res["E2"] = dict(ranking=sorted(zip(names, imp.tolist()), key=lambda t: -t[1]))
vecs = []
for seed in [42, 43, 44]:
    Bb, _, _, _, _, Xte_b = models[("binary", seed)]
    r2 = np.random.default_rng(seed)
    vecs.append(np.abs(shap.TreeExplainer(Bb).shap_values(Xte_b[r2.choice(len(Xte_b), 2000, replace=False)])).mean(0))
res["E2"]["stability"] = [spearmanr(vecs[i], vecs[j]).correlation for i, j in [(0, 1), (0, 2), (1, 2)]]
print("E2 top-5:", [n for n, _ in res["E2"]["ranking"][:5]], "stability", np.round(res["E2"]["stability"], 3))

# ---- E3: withheld subtypes
held = np.isin(sub, WITHHELD)
for seed in SEEDS:
    Xk, yk = X[~held], yb[~held]
    Xtr, Xte, ytr, yte = train_test_split(Xk, yk, test_size=0.3, stratify=yk, random_state=seed)
    B, H, feats, *_ = run_pair(Xtr, ytr, seed, 2)
    row = dict(seed=seed, seen_f1_base=f1_score(yte, B.predict(Xte)), seen_f1_hyb=f1_score(yte, H.predict(feats(Xte))))
    for s in WITHHELD:
        m = sub == s
        row[s] = dict(n=int(m.sum()), base=float(B.predict(X[m]).mean()), hyb=float(H.predict(feats(X[m])).mean()))
    m = held
    row["pooled"] = dict(n=int(m.sum()), base=float(B.predict(X[m]).mean()), hyb=float(H.predict(feats(X[m])).mean()))
    res["E3"].append(row)
    print(f"E3 seed {seed}: pooled base {row['pooled']['base']:.3f} hyb {row['pooled']['hyb']:.3f}", flush=True)

# ---- E4: size and single-thread latency (5-class models, seed 42, 10,000 test records)
B, H, feats, sc, mlp, Xte = models[("5class", 42)]
Xb = Xte[:10000]


def lat(fn, reps=21):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)) / len(Xb) * 1e6
res["E4"] = dict(base_kb=len(pickle.dumps(B)) / 1024, hyb_kb=len(pickle.dumps((sc, mlp, H))) / 1024,
                 base_us=lat(lambda: B.predict(Xb)), hyb_us=lat(lambda: H.predict(feats(Xb))))
print("E4", {k: round(v, 2) for k, v in res["E4"].items()})


# ---- summary: Wilcoxon and win counts
def summ(task, key):
    b = np.array([r["base"][key] for r in res["E1"][task]]); h = np.array([r["hyb"][key] for r in res["E1"][task]])
    diff = h - b
    p = wilcoxon(h, b).pvalue if np.any(diff != 0) else 1.0
    return dict(base=[b.mean(), b.std(ddof=1)], hyb=[h.mean(), h.std(ddof=1)], delta=diff.mean(),
                hyb_wins=int((diff > 0).sum()), p=float(p))
res["summary"] = {f"{t}:{k}": summ(t, k) for t, ks in
                  [("5class", ["acc", "macro_f1", "macro_p", "macro_r", "weighted_f1"]),
                   ("binary", ["acc", "f1", "precision", "recall"])] for k in ks}
json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, default=float)
for k, v in res["summary"].items():
    print(f"{k:20s} base {v['base'][0]:.4f} hyb {v['hyb'][0]:.4f} d {v['delta']:+.4f} wins {v['hyb_wins']} p {v['p']:.4f}")
print("done")
