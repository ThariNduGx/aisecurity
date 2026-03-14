"""
train_model.py  –  Multi-class model comparison on UNSW-NB15
=============================================================
Trains 4 classifiers, compares them, tunes the best one, and saves it.

Labels    : attack_cat  (multi-class: Normal, DoS, Reconnaissance, …)
Algorithms:
  1. Decision Tree Classifier
  2. K-Nearest Neighbors (KNN)
  3. Random Forest Classifier
  4. Support Vector Machine (LinearSVC – fast on large data)

Outputs:
  • Console comparison table  (Accuracy / Precision / Recall / F1)
  • confusion_matrix_<name>.png     per model
  • hyperparameter_tuning.png       (max_depth vs accuracy for Decision Tree)
  • moodle_ai_security_model.pkl    best trained model
  • label_encoder.pkl               maps int predictions ↔ attack_cat string

Dataset:
  Download from Kaggle: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
  Place UNSW_NB15_training-set.csv in the ai_backend/ folder, then run:
      python train_model.py
"""

import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless – no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import LinearSVC
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     confusion_matrix,
                                     classification_report)

warnings.filterwarnings('ignore')

# ─── Config ────────────────────────────────────────────────────────────────────
DATASET     = 'UNSW_NB15_training-set.csv'
MODEL_OUT   = 'moodle_ai_security_model.pkl'
ENCODER_OUT = 'label_encoder.pkl'
LABEL_COL   = 'attack_cat'

# Categorical + non-feature columns to drop.
# 'Label' (binary 0/1) is excluded so only the multi-class column is used.
COLS_TO_DROP = [
    'srcip', 'dstip', 'proto', 'state', 'service',
    'stcpb', 'dtcpb', 'Stime', 'ltime',
    'Label',
]


# ─── Data Loading ──────────────────────────────────────────────────────────────
def load_data():
    print("[INFO] Loading UNSW-NB15 dataset …")
    try:
        df = pd.read_csv(DATASET)
    except FileNotFoundError:
        print("[ERROR] Dataset not found.")
        print("        Download from Kaggle:")
        print("        https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
        print(f"        Place '{DATASET}' in the ai_backend/ folder.")
        return None

    print(f"[INFO] Loaded {len(df):,} rows × {len(df.columns)} columns.")

    # ── Clean attack_cat ──────────────────────────────────────────────────────
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].replace({'nan': 'Normal', '': 'Normal'})
    print(f"\n[INFO] Class distribution (attack_cat):\n{df[LABEL_COL].value_counts()}\n")

    # ── Drop non-feature columns ──────────────────────────────────────────────
    cols_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_drop)

    # ── Build X (numeric features) and y (encoded labels) ────────────────────
    X = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number])
    X = X.fillna(0)

    le = LabelEncoder()
    y  = le.fit_transform(df[LABEL_COL])

    print(f"[INFO] Features  : {X.shape[1]} numeric columns")
    print(f"[INFO] Classes   : {list(le.classes_)}")
    print(f"[INFO] Feature list: {list(X.columns)}\n")

    # Save encoder so app.py can decode predictions at inference time
    joblib.dump(le, ENCODER_OUT)
    print(f"[INFO] LabelEncoder saved → {ENCODER_OUT}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, le, list(X.columns)


# ─── Metrics helper ────────────────────────────────────────────────────────────
def metrics_row(name, y_test, y_pred):
    return {
        'Algorithm' : name,
        'Accuracy'  : round(accuracy_score (y_test, y_pred) * 100, 2),
        'Precision' : round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
        'Recall'    : round(recall_score   (y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
        'F1'        : round(f1_score       (y_test, y_pred, average='weighted', zero_division=0) * 100, 2),
    }


# ─── Confusion Matrix plot ─────────────────────────────────────────────────────
def plot_confusion_matrix(name, y_test, y_pred, class_labels):
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_title(f'Confusion Matrix – {name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fname = f"confusion_matrix_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(fname, dpi=100)
    plt.close()
    print(f"[INFO] Saved → {fname}")


# ─── Hyperparameter tuning (Decision Tree max_depth) ──────────────────────────
def plot_hyperparam_tuning(X_train, y_train, X_test, y_test):
    """
    Sweep max_depth 1–30, record train & test accuracy.
    Plots: Hyperparameter value (max_depth) vs Accuracy.
    Demonstrates overfitting/underfitting and identifies the optimal depth.
    """
    depths    = list(range(1, 31))
    train_acc = []
    test_acc  = []

    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        train_acc.append(accuracy_score(y_train, clf.predict(X_train)) * 100)
        test_acc .append(accuracy_score(y_test,  clf.predict(X_test))  * 100)

    best_depth = depths[int(np.argmax(test_acc))]
    print(f"[INFO] Best max_depth = {best_depth}  "
          f"(test accuracy {max(test_acc):.2f}%)")

    plt.figure(figsize=(10, 5))
    plt.plot(depths, train_acc, 'o-', label='Train Accuracy', color='steelblue')
    plt.plot(depths, test_acc,  's-', label='Test Accuracy',  color='darkorange')
    plt.axvline(best_depth, linestyle='--', color='red',
                label=f'Best depth = {best_depth}')
    plt.xlabel('max_depth  (Hyperparameter)')
    plt.ylabel('Accuracy (%)')
    plt.title('Decision Tree – Hyperparameter Tuning\nmax_depth vs Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('hyperparameter_tuning.png', dpi=100)
    plt.close()
    print("[INFO] Saved → hyperparameter_tuning.png")

    return best_depth


# ─── Main training pipeline ────────────────────────────────────────────────────
def train():
    result = load_data()
    if result is None:
        return

    X_train, X_test, y_train, y_test, le, feature_cols = result

    # ── Define the 4 classifiers ──────────────────────────────────────────────
    classifiers = [
        ('Decision Tree',  DecisionTreeClassifier(random_state=42)),
        ('KNN',            KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        ('Random Forest',  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('SVM (Linear)',   LinearSVC(random_state=42, max_iter=2000)),
    ]

    table   = []   # rows for the comparison table
    trained = {}   # name → (fitted_clf, y_pred)

    for name, clf in classifiers:
        print(f"\n[INFO] Training {name} …")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        row = metrics_row(name, y_test, y_pred)
        table.append(row)
        trained[name] = (clf, y_pred)

        print(f"       Accuracy={row['Accuracy']}%  "
              f"Precision={row['Precision']}%  "
              f"Recall={row['Recall']}%  "
              f"F1={row['F1']}%")
        print(classification_report(y_test, y_pred,
                                    target_names=le.classes_, zero_division=0))
        plot_confusion_matrix(name, y_test, y_pred, le.classes_)

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON TABLE  –  Multi-class on UNSW-NB15")
    print("=" * 65)
    df_table = pd.DataFrame(table).set_index('Algorithm')
    print(df_table.to_string())
    print("=" * 65 + "\n")

    best_name = df_table['F1'].idxmax()
    print(f"[INFO] Best model by weighted F1: {best_name}  "
          f"({df_table.loc[best_name, 'F1']}%)")

    # ── Hyperparameter tuning (always on Decision Tree for the report) ────────
    print("\n[INFO] Hyperparameter tuning for Decision Tree (max_depth) …")
    best_depth = plot_hyperparam_tuning(X_train, y_train, X_test, y_test)

    # Retrain Decision Tree with tuned depth
    tuned_dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    tuned_dt.fit(X_train, y_train)
    tuned_f1 = f1_score(y_test, tuned_dt.predict(X_test),
                        average='weighted', zero_division=0) * 100
    print(f"[INFO] Tuned Decision Tree  F1 = {tuned_f1:.2f}%  "
          f"(best_depth={best_depth})")

    # ── Save the production model ─────────────────────────────────────────────
    # Use the overall best model; if Decision Tree is best, use its tuned version
    if best_name == 'Decision Tree':
        model_to_save = tuned_dt
    else:
        model_to_save = trained[best_name][0]

    joblib.dump(model_to_save, MODEL_OUT)
    print(f"\n[SUCCESS] Model saved  → {MODEL_OUT}  ({best_name})")
    print(f"[SUCCESS] Features ({len(feature_cols)}): {feature_cols}")


if __name__ == '__main__':
    train()
