"""
train_model.py  –  Multi-class model comparison on UNSW-NB15
=============================================================
Mirrors the Google Colab notebook exactly, adapted for standalone execution.

Steps:
  1. Data Preprocessing & Missing Data Handling
  2. Train 4 algorithms: Decision Tree, Random Forest, KNN, SVM (LinearSVC)
  3. Print comparison table: Accuracy / Precision / Recall / F1-Score / Time
  4. Generate per-model confusion matrix PNGs
  5. Hyperparameter tuning graph (max_depth vs Accuracy) for Decision Tree
  6. Export final tuned model + scaler

Outputs:
  moodle_ai_security_model.pkl   – best (tuned Decision Tree) model
  scaler.pkl                     – StandardScaler fitted on training data
  confusion_matrices.png         – 2×2 grid, one matrix per algorithm
  hyperparameter_tuning.png      – max_depth vs Accuracy graph

Dataset:
  https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
  Place UNSW_NB15_training-set.csv in the ai_backend/ folder, then run:
      python train_model.py
"""

import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless – no display needed outside Colab
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import LinearSVC
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     confusion_matrix)

warnings.filterwarnings('ignore')

# ─── 1. Data Preprocessing & Missing Data Handling ─────────────────────────────

print("[INFO] Loading Dataset...")
try:
    df = pd.read_csv('UNSW_NB15_training-set.csv')
except FileNotFoundError:
    print("[ERROR] Dataset not found.")
    print("        Download from Kaggle:")
    print("        https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
    print("        Place 'UNSW_NB15_training-set.csv' in the ai_backend/ folder.")
    raise SystemExit(1)

# Handle missing data
# In UNSW-NB15 normal traffic rows often have a blank attack_cat
df['attack_cat'] = df['attack_cat'].fillna('Normal')
df['attack_cat'] = df['attack_cat'].replace('Backdoor', 'Backdoors')  # Fix dataset typo

# Fill any empty numeric cells with 0 to prevent algorithm crashes
df = df.fillna(0)

# Drop binary label and irrelevant text columns
cols_to_drop  = ['id', 'label', 'proto', 'service', 'state']
cols_present  = [c for c in cols_to_drop if c in df.columns]
df_clean      = df.drop(columns=cols_present)

# Separate features (X) and target (y) – only numeric columns for X
X = df_clean.select_dtypes(include=[np.number])
y = df_clean['attack_cat']

print(f"[INFO] Features mapped: {X.shape[1]} numeric features.")
print(f"[INFO] Classes found  : {sorted(y.unique())}")

# Train/Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling – required for KNN and SVM; applied to all for consistency
scaler          = StandardScaler()
X_train_scaled  = scaler.fit_transform(X_train)
X_test_scaled   = scaler.transform(X_test)

print("[INFO] Preprocessing complete. Data is scaled and split.")
print(f"       Train size: {len(X_train):,}   Test size: {len(X_test):,}\n")


# ─── 2. Train the Four Algorithms ──────────────────────────────────────────────

models = {
    "Decision Tree"          : DecisionTreeClassifier(random_state=42),
    "Random Forest"          : RandomForestClassifier(n_estimators=50, random_state=42),
    "K-Nearest Neighbors"    : KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine" : LinearSVC(random_state=42, max_iter=1000, dual=False),
}

results    = []
trained    = {}   # name → (fitted model, y_pred)

print("[INFO] Training models. Please wait...")

for name, model in models.items():
    start = time.time()

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc  = accuracy_score (y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score   (y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score       (y_test, y_pred, average='weighted', zero_division=0)
    elapsed = round(time.time() - start, 2)

    results.append({
        "Algorithm"  : name,
        "Accuracy"   : round(acc  * 100, 2),
        "Precision"  : round(prec * 100, 2),
        "Recall"     : round(rec  * 100, 2),
        "F1-Score"   : round(f1   * 100, 2),
        "Time (sec)" : elapsed,
    })
    trained[name] = (model, y_pred)

    print(f"[SUCCESS] {name} trained in {elapsed}s  "
          f"Acc={round(acc*100,2)}%  F1={round(f1*100,2)}%")


# ─── 3. Comparison Table ───────────────────────────────────────────────────────

results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("  MODEL COMPARISON TABLE  –  Multi-class on UNSW-NB15")
print("=" * 70)
print(results_df.to_string(index=False))
print("=" * 70 + "\n")


# ─── 4. Confusion Matrices (2×2 grid, one per algorithm) ──────────────────────

class_labels = sorted(y.unique())

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Confusion Matrices for Multi-Class Threat Detection',
             fontsize=20, fontweight='bold')
axes = axes.flatten()

for idx, (name, (model, y_pred)) in enumerate(trained.items()):
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=class_labels, yticklabels=class_labels)
    axes[idx].set_title(f'{name} Confusion Matrix', fontsize=14)
    axes[idx].set_ylabel('Actual Traffic Class')
    axes[idx].set_xlabel('Predicted Traffic Class')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('confusion_matrices.png', dpi=100)
plt.close()
print("[INFO] Saved → confusion_matrices.png")


# ─── 5. Hyperparameter Tuning – Decision Tree max_depth ───────────────────────

train_accuracies = []
test_accuracies  = []
depth_range      = range(1, 26)

print("\n[INFO] Running Hyperparameter Tuning. Testing Tree Depths 1–25 …")

for depth in depth_range:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train_scaled, y_train)
    train_accuracies.append(accuracy_score(y_train, dt.predict(X_train_scaled)))
    test_accuracies .append(accuracy_score(y_test,  dt.predict(X_test_scaled)))

optimal_depth = list(depth_range)[test_accuracies.index(max(test_accuracies))]
print(f"[RESULT] Optimal max_depth = {optimal_depth}  "
      f"(test accuracy {max(test_accuracies)*100:.2f}%)")

plt.figure(figsize=(10, 6))
plt.plot(depth_range, train_accuracies,
         label='Training Accuracy', marker='o', color='blue')
plt.plot(depth_range, test_accuracies,
         label='Testing Accuracy',  marker='s', color='red')
plt.axvline(x=optimal_depth, color='green', linestyle=':',
            label=f'Optimal Depth ({optimal_depth})')
plt.title('Hyperparameter Tuning: Decision Tree Max Depth vs Accuracy', fontsize=16)
plt.xlabel('Hyperparameter Value (max_depth)', fontsize=12)
plt.ylabel('Model Accuracy', fontsize=12)
plt.xticks(np.arange(1, 26, 2))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('hyperparameter_tuning.png', dpi=100)
plt.close()
print("[INFO] Saved → hyperparameter_tuning.png")


# ─── 6. Export Final Tuned Model + Scaler ──────────────────────────────────────

final_model = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
final_model.fit(X_train_scaled, y_train)

joblib.dump(final_model, 'moodle_ai_security_model.pkl')
joblib.dump(scaler,      'scaler.pkl')

print("\n[SUCCESS] Final multi-class model saved  → moodle_ai_security_model.pkl")
print("[SUCCESS] StandardScaler saved           → scaler.pkl")
print("          (Both files must be present when running app.py)")
