"""
rt_duplicates.py - duplicate structure of RT-IoT2022 (Sections 4.1.4 and 4.6.6).

Counts exact duplicate records with and without the port identifiers, the number of distinct
port-free feature vectors per label, the share of test records that have an exact copy in the
training split under the 70/30 splits of Algorithm 3 (seeds 42-46), and whether the four withheld
subtypes of E3 share feature vectors with the records that remain in training.
Reads RT_IOT2022.csv (checksum as in Table 24); writes results_rtiot_duplicates.json.
"""
import hashlib, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

SHA = "d9b315d6d899cd60d356f7d3dbe61907f1163985bd800a87cffe600eb577bd20"
assert hashlib.sha256(open("RT_IOT2022.csv", "rb").read()).hexdigest() == SHA, "checksum mismatch"
r = pd.read_csv("RT_IOT2022.csv")
lab = "Attack_type"
COARSE = {"Thing_Speak": 0, "MQTT_Publish": 0, "Wipro_bulb": 0, "DOS_SYN_Hping": 1, "DDOS_Slowloris": 1,
          "NMAP_UDP_SCAN": 2, "NMAP_XMAS_TREE_SCAN": 2, "NMAP_OS_DETECTION": 2, "NMAP_TCP_scan": 2,
          "NMAP_FIN_SCAN": 2, "Metasploit_Brute_Force_SSH": 3, "ARP_poisioning": 4}
WITHHELD = ["NMAP_XMAS_TREE_SCAN", "NMAP_FIN_SCAN", "DDOS_Slowloris", "Metasploit_Brute_Force_SSH"]
BENIGN = ["Thing_Speak", "MQTT_Publish", "Wipro_bulb"]

# port-free feature vector, exactly as used by the models (rt_iot2022.py)
X = pd.get_dummies(r.drop(columns=["no", "id.orig_p", "id.resp_p", lab]), columns=["proto", "service"], dtype=float)
r["key"] = pd.util.hash_pandas_object(X, index=False).to_numpy()
y5 = r[lab].map(COARSE).to_numpy()
out = {"records": len(r)}

# with ports kept (as in the integrity audit of ref. [24])
withports = r.drop(columns=["no", "key"])
out["duplicate_rows_with_ports_incl_label"] = int(withports.duplicated().sum())
out["duplicate_rows_with_ports_features_only"] = int(withports.drop(columns=[lab]).duplicated().sum())

# without ports (the model input)
out["distinct_portfree_vectors"] = int(r.key.nunique())
out["duplicate_rows_portfree_features"] = int(len(r) - r.key.nunique())
n_lab = r.groupby("key")[lab].nunique()
out["portfree_vectors_with_conflicting_labels"] = int((n_lab > 1).sum())
per = r.groupby(lab).agg(records=("key", "size"), distinct_vectors=("key", "nunique"))
out["per_label"] = {k: dict(records=int(v.records), distinct_vectors=int(v.distinct_vectors))
                    for k, v in per.iterrows()}

# share of test records with an exact copy in the training split (Algorithm 3 splits, 5-class stratification)
share = []
for seed in range(42, 47):
    tr, te = train_test_split(np.arange(len(r)), test_size=0.3, stratify=y5, random_state=seed)
    share.append(float(np.isin(r.key.to_numpy()[te], r.key.to_numpy()[tr]).mean()))
out["test_records_with_exact_copy_in_train"] = share

# E3: do withheld subtypes share feature vectors with the records that stay in training?
rest = r[~r[lab].isin(WITHHELD)]
att = set(rest[~rest[lab].isin(BENIGN)].key); ben = set(rest[rest[lab].isin(BENIGN)].key)
out["E3_overlap"] = {w: dict(records=int((r[lab] == w).sum()),
                             share_identical_to_retained_attack_vector=round(float(r[r[lab] == w].key.isin(att).mean()), 4),
                             share_identical_to_benign_vector=round(float(r[r[lab] == w].key.isin(ben).mean()), 4))
                     for w in WITHHELD}

json.dump(out, open("results_rtiot_duplicates.json", "w"), indent=1)
print(json.dumps(out, indent=1))
