"""
Data pipeline: load -> connection key -> remove duplicates -> split -> scale -> save.
CSE is read batch by batch (all attacks + a fraction of benign rows). The connection key
(IPs, ports, protocol) is used to split by connection, so one connection never lands in
two sets. Heavy-tailed columns (bytes, packets, durations, throughput) get a log1p, then a
scaler is fitted on the train set only. For cross-dataset tests, the target is scaled
with the source scaler (never refitted).
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from . import features as F

ROOT = Path(__file__).resolve().parent.parent
RAW_UNSW = ROOT / "data" / "raw" / "NF-UNSW-NB15-v2.csv"
RAW_CSE = ROOT / "data" / "raw" / "archive" / "NF-CSE-CIC-IDS2018-V2.parquet"
PROCESSED = ROOT / "data" / "processed"
KEY_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL"]
LOG_WORDS = ("BYTES", "PKT", "DURATION", "THROUGHPUT", "TCP_WIN")


def load_unsw(path=RAW_UNSW):
    df = pd.read_csv(path)
    num = [c for c in df.select_dtypes("number").columns if c != "Label"]
    df[num] = df[num].astype("float32")
    return df


def load_cse(path=RAW_CSE, benign_frac=0.05, seed=42):
    import pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    kept = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=500_000):
        df = batch.to_pandas()
        keep = (df["Label"] == 1).values | (rng.random(len(df)) < benign_frac)
        kept.append(df[keep])
    df = pd.concat(kept, ignore_index=True)
    df["Label"] = df["Label"].astype("int8")
    if "Attack" in df.columns:
        df["Attack"] = df["Attack"].astype(str)
    return df


def add_connection_key(df):
    df = df.copy()
    cols = [c for c in KEY_COLS if c in df.columns]
    codes = pd.DataFrame({c: pd.factorize(df[c])[0] for c in cols})
    df["conn_id"] = pd.util.hash_pandas_object(codes, index=False).values
    return df


def remove_duplicates(df):
    cols = [c for c in F.get_feature_columns(df) if c != "L4_SRC_PORT"] + ["Label"]
    clean = df.drop_duplicates(subset=cols).reset_index(drop=True)
    return clean, len(df) - len(clean)


def split_by_connection(df, seed=42):
    g = df["conn_id"].values
    train, rest = next(GroupShuffleSplit(1, train_size=0.70, random_state=seed).split(df, groups=g))
    val, test = next(GroupShuffleSplit(1, train_size=0.50, random_state=seed).split(rest, groups=g[rest]))
    return train, rest[val], rest[test]


def transform(df, features, idx_train, idx_val, idx_test):
    X, y = df[features].to_numpy("float64"), df["Label"].to_numpy("int8")
    X = np.clip(np.nan_to_num(X, nan=0, posinf=0, neginf=0), -1e10, 1e10)
    log_idx = [i for i, c in enumerate(features) if any(w in c for w in LOG_WORDS)]
    X[:, log_idx] = np.log1p(np.clip(X[:, log_idx], 0, None))
    X = X.astype("float32")
    scaler = StandardScaler().fit(X[idx_train])
    out = {}
    for name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        out["X_" + name], out["y_" + name] = scaler.transform(X[idx]).astype("float32"), y[idx]
    return out, scaler


def save_processed(name, arrays, scaler, idx_train, idx_val, idx_test):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED / f"{name}.npz", **arrays,
                        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)
    joblib.dump(scaler, PROCESSED / f"{name}_scaler.joblib")


def _cols(feature_set):
    sets = F.load_feature_sets()
    return [sets["all"].index(c) for c in sets[feature_set]]


def load_dataset(name, feature_set="all"):
    d, cols = np.load(PROCESSED / f"{name}.npz"), _cols(feature_set)
    return (d["X_train"][:, cols], d["y_train"], d["X_val"][:, cols],
            d["y_val"], d["X_test"][:, cols], d["y_test"])


def load_cross(source, target, feature_set="all"):
    X_train, y_train, X_val, y_val, _, _ = load_dataset(source, feature_set)
    tgt = np.load(PROCESSED / f"{target}.npz")
    sc_src = joblib.load(PROCESSED / f"{source}_scaler.joblib")
    sc_tgt = joblib.load(PROCESSED / f"{target}_scaler.joblib")
    X_tgt = sc_src.transform(sc_tgt.inverse_transform(tgt["X_test"])).astype("float32")
    return X_train, y_train, X_val, y_val, X_tgt[:, _cols(feature_set)], tgt["y_test"]