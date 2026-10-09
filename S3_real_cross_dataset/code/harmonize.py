"""Harmonize NSL-KDD, UNSW-NB15 and RT-IoT2022 to one shared flow schema and a
shared three-class label space (Benign, DoS/DDoS, Reconnaissance).

Every feature is computed from fields that exist in all three releases; no
field that is unique to one dataset is used.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

DATA = os.environ.get("XFCA_DATA", str(Path(__file__).resolve().parent.parent / "data"))
CLASSES = ["Benign", "DoS/DDoS", "Reconnaissance"]
DOMAINS = ["NSL-KDD", "UNSW-NB15", "RT-IoT2022"]

NSL_DOS = {"back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
           "mailbomb", "processtable", "udpstorm"}
NSL_PROBE = {"ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"}

# service -> harmonized service group
SVC = {
    # NSL-KDD
    "http": "http", "http_8001": "http", "http_2784": "http", "www": "http",
    "http_443": "ssl", "domain": "dns", "domain_u": "dns",
    "ftp": "ftp", "ftp_data": "ftp", "ssh": "ssh",
    "smtp": "mail", "pop_2": "mail", "pop_3": "mail", "imap4": "mail",
    "private": "unknown", "other": "unknown", "eco_i": "unknown",
    "ecr_i": "unknown", "urp_i": "unknown", "urh_i": "unknown",
    "red_i": "unknown", "tim_i": "unknown",
    # UNSW-NB15 / RT-IoT2022
    "-": "unknown", "dns": "dns", "ftp-data": "ftp", "pop3": "mail",
    "ssl": "ssl",
}
SVC_GROUPS = ["http", "dns", "ftp", "ssh", "mail", "ssl", "other", "unknown"]

FEATURES = (["log_duration", "log_src_bytes", "log_dst_bytes", "dst_byte_ratio",
             "log_byte_rate", "src_bytes_zero", "dst_bytes_zero",
             "fin_closed", "rst_seen", "proto_tcp", "proto_udp", "proto_icmp"]
            + [f"svc_{g}" for g in SVC_GROUPS])


def _common(dur, sb, db, proto, svc, fin, rst):
    dur = np.clip(np.asarray(dur, float), 0, None)
    sb = np.clip(np.asarray(sb, float), 0, None)
    db = np.clip(np.asarray(db, float), 0, None)
    out = pd.DataFrame({
        "log_duration": np.log1p(dur),
        "log_src_bytes": np.log1p(sb),
        "log_dst_bytes": np.log1p(db),
        "dst_byte_ratio": db / np.maximum(sb + db, 1.0),
        "log_byte_rate": np.log1p((sb + db) / np.maximum(dur, 1e-3)),
        "src_bytes_zero": (sb == 0).astype(float),
        "dst_bytes_zero": (db == 0).astype(float),
        "fin_closed": np.asarray(fin, float),
        "rst_seen": np.asarray(rst, float),
    })
    proto = pd.Series(proto).str.lower().values
    for p in ["tcp", "udp", "icmp"]:
        out[f"proto_{p}"] = (proto == p).astype(float)
    grp = pd.Series(svc).map(lambda s: SVC.get(s, "other")).values
    for g in SVC_GROUPS:
        out[f"svc_{g}"] = (grp == g).astype(float)
    return out[FEATURES]


def load_nsl():
    cols = (["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"]
            + [f"c{i}" for i in range(35)] + ["label", "difficulty"])
    d = pd.concat([pd.read_csv(f"{DATA}/{f}", header=None, names=cols)
                   for f in ["KDDTrain+.txt", "KDDTest+.txt"]], ignore_index=True)
    y = np.where(d.label == "normal", 0, np.where(d.label.isin(NSL_DOS), 1,
                 np.where(d.label.isin(NSL_PROBE), 2, -1)))
    fin = (d.flag == "SF") & (d.protocol_type == "tcp")
    rst = d.flag.isin(["REJ", "RSTO", "RSTR", "RSTOS0"])
    X = _common(d.duration, d.src_bytes, d.dst_bytes, d.protocol_type, d.service, fin, rst)
    return X, y, d.label.values


def load_unsw():
    d = pd.concat([pd.read_csv(f"{DATA}/UNSW_NB15_training-set.csv"),
                   pd.read_csv(f"{DATA}/UNSW_NB15_testing-set.csv")], ignore_index=True)
    m = {"Normal": 0, "DoS": 1, "Reconnaissance": 2}
    y = d.attack_cat.map(m).fillna(-1).astype(int).values
    X = _common(d.dur, d.sbytes, d.dbytes, d.proto, d.service,
                d.state == "FIN", d.state == "RST")
    return X, y, d.attack_cat.values


def load_rt():
    d = pd.read_csv(f"{DATA}/RT_IOT2022.csv")
    a = d.Attack_type
    y = np.where(a.isin(["Thing_Speak", "MQTT_Publish", "Wipro_bulb"]), 0,
                 np.where(a.isin(["DOS_SYN_Hping", "DDOS_Slowloris"]), 1,
                          np.where(a.str.startswith("NMAP"), 2, -1)))
    fin = (d.proto == "tcp") & (d.flow_FIN_flag_count > 0)
    X = _common(d.flow_duration, d["fwd_pkts_payload.tot"], d["bwd_pkts_payload.tot"],
                d.proto, d.service, fin, d.flow_RST_flag_count > 0)
    return X, y, a.values


def load_all():
    out = {}
    for name, fn in zip(DOMAINS, [load_nsl, load_unsw, load_rt]):
        X, y, raw = fn()
        keep = y >= 0
        out[name] = (X[keep].reset_index(drop=True).values.astype(np.float64),
                     y[keep], raw[keep], int((~keep).sum()), len(y))
    return out


if __name__ == "__main__":
    for k, (X, y, raw, dropped, total) in load_all().items():
        print(k, X.shape, "dropped", dropped, "of", total,
              {CLASSES[c]: int((y == c).sum()) for c in range(3)})
