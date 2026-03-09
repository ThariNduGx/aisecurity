"""
train_model.py – Train a Decision Tree classifier on the UNSW-NB15 dataset,
                  replicating the Google Colab model used in this project.

Dataset:
  Download from Kaggle: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
  Place UNSW_NB15_training-set.csv in the ai_backend/ folder, then run:
      python train_model.py

Output:
  moodle_app_security_model.pkl  – the 39-feature model loaded by app.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Columns dropped before training (categorical strings + non-feature columns).
# proto, state, service are dropped because they are categorical strings.
# stcpb, dtcpb, Stime are dropped as they are large sequence numbers / timestamps.
COLS_TO_DROP = [
    'srcip', 'dstip', 'proto', 'state', 'service',
    'stcpb', 'dtcpb', 'Stime', 'attack_cat',
]

LABEL_COL = 'Label'

# The remaining 39 columns are all numeric — no encoding is applied.
# X_train_numeric = df.drop(COLS_TO_DROP + [LABEL_COL], axis=1).select_dtypes(include=[np.number])


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

    # Drop categorical and non-feature columns; keep only numeric columns.
    # No encoding is applied — proto, state, service are completely removed.
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_present)

    X = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number])
    y = df[LABEL_COL]

    print(f"[INFO] Training on {X.shape[1]} numeric features (X_train_numeric).")
    print(f"[INFO] Feature columns: {list(X.columns)}")

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

    joblib.dump(model, 'moodle_app_security_model.pkl')
    print("[SUCCESS] Model saved as moodle_app_security_model.pkl")


if __name__ == '__main__':
    train()
