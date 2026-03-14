"""
simulate_attack.py  –  Attack Simulation Tool
==============================================
Loads real samples from UNSW-NB15 and sends them to the Flask API to
demonstrate that the trained model correctly identifies attack vs normal.

Features are sent RAW (unscaled) – app.py scales them with scaler.pkl
before prediction, exactly as done during training.

Usage:
    python simulate_attack.py                         # 5 normal + 5 attack
    python simulate_attack.py --samples 20            # 10 normal + 10 attack
    python simulate_attack.py --url http://IP:5001    # custom API URL
"""

import argparse
import json
import time
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

API_URL  = 'http://127.0.0.1:5001/predict'
DATASET  = 'UNSW_NB15_training-set.csv'

# Must match exactly what train_model.py drops
COLS_TO_DROP = ['id', 'label', 'proto', 'service', 'state']
LABEL_COL    = 'attack_cat'


def load_samples(n_each: int):
    print(f"[INFO] Loading {DATASET} …")
    try:
        df = pd.read_csv(DATASET)
    except FileNotFoundError:
        print(f"[ERROR] {DATASET} not found in ai_backend/.")
        return None

    # Same cleaning as train_model.py
    df[LABEL_COL] = df[LABEL_COL].fillna('Normal')
    df[LABEL_COL] = df[LABEL_COL].replace('Backdoor', 'Backdoors')
    df = df.fillna(0)

    cols_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_drop)

    X = df.select_dtypes(include=[np.number])
    y = df[LABEL_COL]

    normal_idx = y[y == 'Normal'].index
    attack_idx = y[y != 'Normal'].index

    normal = X.loc[normal_idx].sample(n=min(n_each, len(normal_idx)), random_state=42)
    attack = X.loc[attack_idx].sample(n=min(n_each, len(attack_idx)), random_state=42)

    return (normal.values, y[normal.index].tolist(),
            attack.values, y[attack.index].tolist(),
            list(X.columns))


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
        return {'error': f'HTTP {e.code}: {e.read().decode()}'}
    except Exception as e:
        return {'error': str(e)}


def run(n_each: int, url: str):
    result = load_samples(n_each)
    if result is None:
        return

    normal_X, normal_y, attack_X, attack_y, feature_cols = result
    print(f"[INFO] Feature columns ({len(feature_cols)}): {feature_cols}\n")

    correct = 0
    total   = 0

    print(f"── NORMAL SAMPLES ({len(normal_X)}) ──────────────────────────────────")
    for i, (row, true_label) in enumerate(zip(normal_X, normal_y)):
        resp       = call_api(row.tolist(), user_id=i + 100, ip='192.168.1.10', url=url)
        prediction = resp.get('prediction', '?')
        status     = resp.get('status', '?')
        ok         = '✓' if status == 'safe' else '✗ WRONG'
        if status == 'safe':
            correct += 1
        total += 1
        print(f"  [{i+1:02d}] true={true_label:20s}  predicted={prediction:20s}  {ok}")
        time.sleep(0.05)

    print(f"\n── ATTACK SAMPLES ({len(attack_X)}) ──────────────────────────────────")
    for i, (row, true_label) in enumerate(zip(attack_X, attack_y)):
        resp       = call_api(row.tolist(), user_id=i + 200, ip='10.0.0.99', url=url)
        prediction = resp.get('prediction', '?')
        status     = resp.get('status', '?')
        ok         = '✓' if status == 'attack' else '✗ MISSED'
        if status == 'attack':
            correct += 1
        total += 1
        print(f"  [{i+1:02d}] true={true_label:20s}  predicted={prediction:20s}  {ok}")
        time.sleep(0.05)

    print(f"\n[RESULT] {correct}/{total} correct  "
          f"({correct/total*100:.1f}% simulation accuracy)")
    print("[DONE]   Check Moodle – attack sessions should be blocked.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UNSW-NB15 Attack Simulator')
    parser.add_argument('--samples', type=int, default=5,
                        help='Normal AND attack samples each (default 5)')
    parser.add_argument('--url', type=str, default=API_URL,
                        help=f'Flask API URL (default {API_URL})')
    args = parser.parse_args()
    run(args.samples, args.url)
