"""
simulate_attack.py  –  Attack Simulation Tool
==============================================
Loads real samples from the UNSW-NB15 dataset and sends them to the
Flask API (/predict) to demonstrate that the trained model correctly
identifies attack vs. normal traffic.

Usage:
    python simulate_attack.py                         # 5 normal + 5 attack
    python simulate_attack.py --samples 20            # 10 normal + 10 attack
    python simulate_attack.py --url http://IP:5001    # custom API URL

This replaces the old odd/even userid trick with genuine traffic signatures
from the dataset, as required by the supervisor.
"""

import argparse
import json
import time
import urllib.request
import urllib.error

import joblib
import numpy as np
import pandas as pd

API_URL  = 'http://127.0.0.1:5001/predict'
DATASET  = 'UNSW_NB15_training-set.csv'

COLS_TO_DROP = [
    'srcip', 'dstip', 'proto', 'state', 'service',
    'stcpb', 'dtcpb', 'Stime', 'ltime',
    'Label',
]
LABEL_COL = 'attack_cat'


def load_samples(n_each: int):
    """Return n_each Normal rows and n_each Attack rows from the dataset."""
    print(f"[INFO] Loading {DATASET} …")
    try:
        df = pd.read_csv(DATASET)
    except FileNotFoundError:
        print(f"[ERROR] {DATASET} not found. Place it in the ai_backend/ folder.")
        return None, None, None

    # Clean
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].replace({'nan': 'Normal', '': 'Normal'})

    cols_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_drop)

    X = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number]).fillna(0)
    y = df[LABEL_COL]

    normal = X[y == 'Normal'].sample(n=min(n_each, (y == 'Normal').sum()),
                                      random_state=42)
    attack = X[y != 'Normal'].sample(n=min(n_each, (y != 'Normal').sum()),
                                      random_state=42)
    labels_n = y[normal.index].tolist()
    labels_a = y[attack.index].tolist()

    return normal.values, labels_n, attack.values, labels_a, list(X.columns)


def call_api(features: list, user_id: int, ip: str, url: str) -> dict:
    payload = json.dumps({
        'userid'  : user_id,
        'ip'      : ip,
        'features': features,
    }).encode('utf-8')

    req = urllib.request.Request(
        url,
        data    = payload,
        headers = {'Content-Type': 'application/json'},
        method  = 'POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return {'error': f'HTTP {e.code}: {body}'}
    except Exception as e:
        return {'error': str(e)}


def run(n_each: int, url: str):
    result = load_samples(n_each)
    if result is None:
        return
    normal_X, normal_y, attack_X, attack_y, feature_cols = result

    print(f"\n[INFO] Sending {len(normal_X)} NORMAL samples …\n")
    for i, (row, true_label) in enumerate(zip(normal_X, normal_y)):
        resp = call_api(row.tolist(), user_id=i + 100, ip='192.168.1.10', url=url)
        status     = resp.get('status', '?')
        attack_cat = resp.get('attack_cat', '?')
        correct    = '✓' if status == 'safe' else '✗ WRONG'
        print(f"  Sample {i+1:02d}  true={true_label:15s}  "
              f"predicted={attack_cat:15s}  status={status}  {correct}")
        time.sleep(0.05)

    print(f"\n[INFO] Sending {len(attack_X)} ATTACK samples …\n")
    for i, (row, true_label) in enumerate(zip(attack_X, attack_y)):
        resp = call_api(row.tolist(), user_id=i + 200, ip='10.0.0.99', url=url)
        status     = resp.get('status', '?')
        attack_cat = resp.get('attack_cat', '?')
        correct    = '✓' if status == 'attack' else '✗ MISSED'
        print(f"  Sample {i+1:02d}  true={true_label:15s}  "
              f"predicted={attack_cat:15s}  status={status}  {correct}")
        time.sleep(0.05)

    print("\n[DONE] Simulation complete.")
    print("       Check the Moodle UI — attack sessions should be blocked.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UNSW-NB15 Attack Simulator')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of normal AND attack samples each (default 5)')
    parser.add_argument('--url', type=str, default=API_URL,
                        help=f'Flask API URL (default {API_URL})')
    args = parser.parse_args()
    run(args.samples, args.url)
