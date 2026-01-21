"""Quick script to train only GradientBoosting with optimized parameters."""
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import uniform
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Constants
DATA_DIR = 'data'
RANDOM_STATE = 42
TEST_SIZE = 0.2
TUNE_SAMPLE_SIZE = 30000
FAST_CV_FOLDS = 3
N_JOBS = -1

ACTIVITY_NAMES = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F',
    7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L',
    13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R'
}

METADATA_COLS = ['subject_id', 'activity_label', 'activity_name', 'sensor_type', 'device']

print("[INFO] Loading data...")
input_path = os.path.join(DATA_DIR, '05_selected', 'minmax_selected.csv')
selected_df = pd.read_csv(input_path)

feature_cols = [c for c in selected_df.columns if c not in METADATA_COLS]
X = selected_df[feature_cols].values
y = selected_df['activity_label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
print(f"Features: {len(feature_cols)}")

# Subsample for tuning
idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), TUNE_SAMPLE_SIZE, replace=False)
X_tune = X_train[idx]
y_tune = y_train[idx]

print(f"\n[INFO] Optimizing GradientBoosting...")
print(f"  Tuning on {len(X_tune):,} samples...")

# Optimized GradientBoosting parameters
model = GradientBoostingClassifier(random_state=RANDOM_STATE)
params = {
    'n_estimators': [50, 100],
    'learning_rate': uniform(0.08, 0.07),  # 0.08-0.15
    'max_depth': [3, 5],
    'subsample': uniform(0.8, 0.2)  # 0.8-1.0
}

start_time = time.time()
random_search = RandomizedSearchCV(
    model, params, n_iter=8, cv=FAST_CV_FOLDS,
    scoring='accuracy', n_jobs=N_JOBS, random_state=RANDOM_STATE, verbose=2
)

random_search.fit(X_tune, y_tune)
tune_time = time.time() - start_time

best_params = random_search.best_params_
cv_score = random_search.best_score_

print(f"  Best params: {best_params}")
print(f"  Tuning CV score: {cv_score*100:.2f}%")
print(f"  Tuning time: {tune_time:.0f}s ({tune_time/60:.1f}m)")
print(f"  Retraining on full {len(X_train):,} samples...")

# Retrain on full dataset
start_time = time.time()
final_model = GradientBoostingClassifier(**best_params, random_state=RANDOM_STATE)
final_model.fit(X_train, y_train)
train_time = time.time() - start_time

print(f"  Full training time: {train_time:.0f}s ({train_time/60:.1f}m)")

# Test
y_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred) * 100

print(f"  Test accuracy: {test_accuracy:.2f}%")

# Save model
output_dir = os.path.join(DATA_DIR, '06b_optimized_results')
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, 'GradientBoosting_optimized.pkl')
joblib.dump(final_model, model_path)
print(f"\n[SAVED] Model saved to {model_path}")

# Generate confusion matrix
vis_dir = 'visualizations/optimized_confusion_matrices'
os.makedirs(vis_dir, exist_ok=True)

labels = sorted(set(y_test) | set(y_pred))
label_names = [ACTIVITY_NAMES.get(l, str(l)) for l in labels]
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=label_names, yticklabels=label_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Optimized GradientBoosting\nAccuracy: {test_accuracy:.2f}%')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'optimized_GradientBoosting_cm.png'), dpi=300)
plt.close()

print(f"[SAVED] Confusion matrix saved to {vis_dir}")

# Update results file
results_path = os.path.join(output_dir, 'optimized_results.csv')
if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)
    # Add or update GB row
    gb_row = pd.DataFrame([{
        'model': 'GradientBoosting',
        'cv_accuracy': cv_score * 100,
        'test_accuracy': test_accuracy,
        'best_params': str(best_params)
    }])
    results_df = pd.concat([results_df[results_df['model'] != 'GradientBoosting'], gb_row], ignore_index=True)
    results_df = results_df.sort_values('test_accuracy', ascending=False)
    results_df.to_csv(results_path, index=False)
    print(f"[SAVED] Updated results to {results_path}")
    print("\nAll Model Results:")
    print(results_df.to_string(index=False))

print("\n[DONE] GradientBoosting training complete!")
