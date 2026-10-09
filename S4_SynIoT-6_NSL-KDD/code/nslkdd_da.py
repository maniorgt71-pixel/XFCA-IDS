"""NSL-KDD re-run of the domain-adversarial-only XFCA variant (Section 4.4).

Source domain: KDDTrain+ (labelled). Target domain: KDDTest+ (labels never used for training;
its records enter only the domain branch, transductive unsupervised domain adaptation).
Conditions: RAW (XGBoost on raw features), BASE (hybrid, lambda = 0), DA (hybrid, lambda = 0.3 fixed).
"""
import json, time
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("XFCA_RESULTS", ROOT / "results"))
DATA = os.environ.get("XFCA_DATA", str(ROOT / "data"))
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score
import xfca_core as xc

CLASSES = ["Benign", "DoS/DDoS", "Reconnaissance", "BruteForce", "WebInject"]
DOS = {"back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "mailbomb", "processtable", "udpstorm"}
PROBE = {"ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"}
R2L = {"ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster",
       "snmpgetattack", "snmpguess", "named", "sendmail", "xlock", "xsnoop", "worm"}
U2R = {"buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel", "ps", "sqlattack", "xterm"}
COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]


def label(l):
    return 0 if l == "normal" else 1 if l in DOS else 2 if l in PROBE else 3 if l in R2L else 4 if l in U2R else -1


def load():
    tr = pd.read_csv(f"{DATA}/KDDTrain+.txt", header=None, names=COLS)
    te = pd.read_csv(f"{DATA}/KDDTest+.txt", header=None, names=COLS)
    ytr, yte = tr.label.map(label).values, te.label.map(label).values
    assert (ytr >= 0).all() and (yte >= 0).all()
    num = [c for c in COLS if c not in ("protocol_type", "service", "flag", "label", "difficulty")]
    def feats(d):
        X = d[num].astype(float).copy()
        for c in ("duration", "src_bytes", "dst_bytes", "hot", "num_compromised", "num_root", "count", "srv_count",
                  "dst_host_count", "dst_host_srv_count"):
            X[c] = np.log1p(X[c])
        for cat in ("protocol_type", "service", "flag"):
            for v in sorted(tr[cat].unique()):            # categories of the training file only
                X[f"{cat}={v}"] = (d[cat] == v).astype(float)
        return X
    Xtr, Xte = feats(tr), feats(te)
    unseen = sorted(set(te.label) - set(tr.label))
    return Xtr, Xte, ytr, yte, list(Xtr.columns), te.label.values, unseen


def run(seeds=(42, 43, 44, 45, 46)):
    Xtr_df, Xte_df, ytr, yte, names, te_sub, unseen = load()
    mu, sd = Xtr_df.values.mean(0), Xtr_df.values.std(0) + 1e-8      # standardization fitted on KDDTrain+ only
    Xtr, Xte = (Xtr_df.values - mu) / sd, (Xte_df.values - mu) / sd
    Xall = np.vstack([Xtr, Xte]); dom = np.r_[np.zeros(len(Xtr), int), np.ones(len(Xte), int)]
    yall = np.r_[ytr, np.zeros(len(Xte), int)]; cmask = dom == 0           # test labels never used
    C = len(CLASSES); out = []; t0 = time.time()
    for seed in seeds:
        raw = xgb.XGBClassifier(**xc.XGB_FINAL, random_state=seed).fit(Xtr, ytr)
        gain = raw.get_booster().get_score(importance_type="gain")
        top3 = sorted(range(len(names)), key=lambda j: -gain.get(f"f{j}", 0))[:3]
        rec = dict(seed=seed, raw_subset=[names[j] for j in top3])
        p = raw.predict(Xte)
        rec["RAW"] = dict(acc=float(accuracy_score(yte, p)), macro_f1=float(f1_score(yte, p, average="macro")),
                          pcf1=f1_score(yte, p, average=None, labels=list(range(C)), zero_division=0).tolist(),
                          size_kb=xc.size_kb(raw))
        for mode in ("BASE", "DA"):
            P, final, _ = xc.train_hybrid("BASE" if mode == "BASE" else "FIXED", Xall, yall, dom, C, 2, seed,
                                          shap_idx=None, cmask=cmask, raw_idx=top3)
            p = final.predict(xc.decision_input(P, Xte, top3))
            rec[mode] = dict(acc=float(accuracy_score(yte, p)), macro_f1=float(f1_score(yte, p, average="macro")),
                             pcf1=f1_score(yte, p, average=None, labels=list(range(C)), zero_division=0).tolist(),
                             size_kb=xc.size_kb(final),
                             unseen_recall=float(np.mean(p[np.isin(te_sub, unseen)] != 0)))
        out.append(rec)
        print(seed, rec["raw_subset"], {m: (round(rec[m]["macro_f1"], 3), round(rec[m]["acc"], 3)) for m in ("RAW", "BASE", "DA")},
              f"({time.time()-t0:.0f}s)", flush=True)
    counts = dict(train={CLASSES[c]: int((ytr == c).sum()) for c in range(C)},
                  test={CLASSES[c]: int((yte == c).sum()) for c in range(C)},
                  n_features=len(names), unseen_subtypes=unseen,
                  unseen_records=int(np.isin(te_sub, unseen).sum()))
    return out, counts


if __name__ == "__main__":
    res, counts = run()
    print(counts)
    json.dump(dict(cfg=xc.CFG, counts=counts, results=res), open(RESULTS / "results_nslkdd.json", "w"), indent=1)
    print("saved")
