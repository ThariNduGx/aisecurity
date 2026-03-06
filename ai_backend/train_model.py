"""
train_model.py – Train a Decision Tree classifier on the UNSW-NB15 dataset,
                  replicating the Google Colab model used in this project.

Dataset:
  Download from Kaggle: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
  Place UNSW_NB15_training-set.csv in the ai_backend/ folder, then run:
      python train_model.py

Output:
  moodle_ai_security_model.pkl  – the 39-feature model loaded by app.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ── 39 feature columns (UNSW-NB15 order, after dropping non-feature columns) ─
FEATURE_COLS = [
    'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
    'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
    'swin', 'dwin', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
    'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
    'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
    'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
]

LABEL_COL = 'Label'

# Columns that are categorical strings and must be label-encoded
CAT_COLS = ['proto', 'service', 'state']


def train():
    print("[INFO] Loading UNSW-NB15 dataset...")
    try:
        df = pd.read_csv('UNSW_NB15_training-set.csv')
    except FileNotFoundError:
        print("[ERROR] Dataset not found.")
        print("        Download from Kaggle:")
        print("        https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
        print("        Place 'UNSW_NB15_training-set.csv' in the ai_backend/ folder.")
        return

    print(f"[INFO] Dataset loaded: {len(df)} rows, {len(df.columns)} columns.")

    # Encode categorical columns using LabelEncoder (alphabetical ordering)
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"[INFO] Encoded '{col}' – unique classes: {list(le.classes_)}")

    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    print(f"[INFO] Feature shape: {X.shape}  |  Label distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("[INFO] Training Decision Tree Classifier...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'moodle_ai_security_model.pkl')
    print("[SUCCESS] Model saved as moodle_ai_security_model.pkl")


if __name__ == '__main__':
    train()
