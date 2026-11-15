"""
Feature lists shared by all models.
Label is the target (0 = benign, 1 = attack), Attack is the attack name (EDA only),
and IP addresses are never used as inputs. The ablation removes ports, then ports + TTL.
"""
import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
IP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
PORT_COLS = ["L4_SRC_PORT", "L4_DST_PORT"]
TTL_COLS = ["MIN_TTL", "MAX_TTL"]


def get_feature_columns(df):
    return [c for c in df.columns if c not in ["Label", "Attack", "conn_id"] + IP_COLS]


def make_feature_sets(features):
    no_ports = [c for c in features if c not in PORT_COLS]
    return {"all": features, "no_ports": no_ports,
            "no_ports_ttl": [c for c in no_ports if c not in TTL_COLS]}


def save_feature_sets(sets):
    json.dump(sets, open(PROCESSED_DIR / "features.json", "w"), indent=2)


def load_feature_sets():
    return json.load(open(PROCESSED_DIR / "features.json"))
