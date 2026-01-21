"""
WISDM-51 Activity Recognition Pipeline
Step 6c: Ensemble Model Training

Combines multiple models using Voting and Stacking for improved accuracy.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                             VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from tqdm import tqdm

from config import (DATA_DIR, VIS_DIR, METADATA_COLS, RANDOM_STATE, TEST_SIZE,
                    CV_FOLDS, N_JOBS, VOTING_WEIGHTS, STACKING_CV, ACTIVITY_NAMES)
from logger import logger

# Ensemble-specific optimizations
ENSEMBLE_CV_SAMPLE_SIZE = 30000
ENSEMBLE_CV_FOLDS = 3
ENSEMBLE_N_JOBS = max(1, (N_JOBS if N_JOBS > 0 else os.cpu_count()) // 2)


def get_base_models():
    """Return optimized base models for ensemble (without GradientBoosting)."""
    
    # REMOVED GradientBoosting - too slow for ensemble training
    models = [
        ('knn', KNeighborsClassifier(
            n_neighbors=5, weights='distance', metric='manhattan',
            algorithm='ball_tree', leaf_size=40
        )),
        ('dt', DecisionTreeClassifier(
            max_depth=25, min_samples_split=4, min_samples_leaf=1,
            criterion='entropy', random_state=RANDOM_STATE
        )),
        ('rf', RandomForestClassifier(
            n_estimators=200, max_depth=25, max_features='sqrt',
            min_samples_split=2, min_samples_leaf=1,
            n_jobs=ENSEMBLE_N_JOBS, random_state=RANDOM_STATE
        ))
    ]
    
    return models


def create_voting_ensemble(models, voting='soft'):
    """Create a voting ensemble classifier."""
    
    # Use only probabilistic models for soft voting (exclude KNN with uniform weights)
    if voting == 'soft':
        # KNN with distance weights supports predict_proba
        pass
    
    return VotingClassifier(
        estimators=models,
        voting=voting,
        n_jobs=ENSEMBLE_N_JOBS
    )


def create_stacking_ensemble(models):
    """Create a stacking ensemble classifier."""
    
    return StackingClassifier(
        estimators=models,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=ENSEMBLE_CV_FOLDS,
        n_jobs=ENSEMBLE_N_JOBS,
        passthrough=False
    )


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """Train model and return evaluation metrics with optimized CV."""
    logger.log(f"  Training {model_name}...")
    
    # Subsample for CV to speed up evaluation
    cv_sample_size = min(ENSEMBLE_CV_SAMPLE_SIZE, len(X_train))
    if cv_sample_size < len(X_train):
        logger.log(f"    Running CV on {cv_sample_size:,} samples (subsampled)...")
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train), cv_sample_size, replace=False)
        X_cv = X_train[idx]
        y_cv = y_train[idx]
    else:
        X_cv = X_train
        y_cv = y_train
    
    # Cross-validation score with reduced folds
    cv = StratifiedKFold(n_splits=ENSEMBLE_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        model, X_cv, y_cv, cv=cv, scoring='accuracy',
        n_jobs=ENSEMBLE_N_JOBS, verbose=0
    )
    cv_mean = cv_scores.mean() * 100
    cv_std = cv_scores.std() * 100
    
    logger.log(f"    CV accuracy: {cv_mean:.2f}% (+/- {cv_std:.2f}%)")
    
    # Train on full training set
    logger.log(f"    Training on full {len(X_train):,} samples...")
    model.fit(X_train, y_train)
    
    # Test accuracy
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred) * 100
    
    logger.log(f"    Test accuracy: {test_accuracy:.2f}%")
    
    return model, y_pred, cv_mean, cv_std, test_accuracy


def generate_ensemble_confusion_matrix(y_true, y_pred, model_name, accuracy):
    """Generate confusion matrix for ensemble model."""
    vis_dir = os.path.join(VIS_DIR, 'ensemble_confusion_matrices')
    os.makedirs(vis_dir, exist_ok=True)
    
    labels = sorted(set(y_true) | set(y_pred))
    label_names = [ACTIVITY_NAMES.get(l, str(l)) for l in labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{model_name}\nAccuracy: {accuracy:.2f}%')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{model_name.replace(" ", "_")}_cm.png'), dpi=300)
    plt.close()


def generate_ensemble_comparison(results_df, baseline_accuracy=None):
    """Generate comparison chart of ensemble models."""
    vis_dir = VIS_DIR
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = results_df['model'].values
    test_acc = results_df['test_accuracy'].values
    
    colors = ['forestgreen' if 'Stacking' in m else 'steelblue' if 'Voting' in m else 'coral' 
              for m in models]
    
    bars = ax.barh(models, test_acc, color=colors)
    
    for bar, val in zip(bars, test_acc):
        ax.annotate(f'{val:.2f}%',
                   xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   xytext=(3, 0), textcoords="offset points",
                   ha='left', va='center', fontsize=10, fontweight='bold')
    
    if baseline_accuracy:
        ax.axvline(x=baseline_accuracy, color='red', linestyle='--', linewidth=2, 
                   label=f'Baseline: {baseline_accuracy:.2f}%')
        ax.legend()
    
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Ensemble Model Performance')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'ensemble_comparison.png'), dpi=300)
    plt.close()
    
    logger.log("Saved ensemble comparison chart")


def run(selected_df=None):
    """Execute Step 6c: Ensemble model training."""
    logger.header("STEP 6C: Ensemble Model Training")
    
    # Load combined selected features (minmax scaled)
    if selected_df is None:
        input_path = os.path.join(DATA_DIR, '05_selected', 'minmax_selected.csv')
        logger.log(f"Loading features from {input_path}")
        selected_df = pd.read_csv(input_path)
    
    feature_cols = [c for c in selected_df.columns if c not in METADATA_COLS]
    
    X = selected_df[feature_cols].values
    y = selected_df['activity_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.log(f"Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")
    logger.log(f"Features: {len(feature_cols)}")
    
    output_dir = os.path.join(DATA_DIR, '06c_ensemble_results')
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # Get base models
    base_models = get_base_models()
    
    # 1. Voting Ensemble (Hard)
    logger.log("\n--- Hard Voting Ensemble ---")
    voting_hard = create_voting_ensemble(base_models, voting='hard')
    voting_hard_trained, y_pred_vh, cv_vh, cv_std_vh, acc_vh = train_and_evaluate(
        voting_hard, "Voting (Hard)", X_train, X_test, y_train, y_test
    )
    generate_ensemble_confusion_matrix(y_test, y_pred_vh, "Voting_Hard", acc_vh)
    joblib.dump(voting_hard_trained, os.path.join(output_dir, 'voting_hard.pkl'))
    results.append({
        'model': 'Voting (Hard)',
        'cv_accuracy': cv_vh,
        'cv_std': cv_std_vh,
        'test_accuracy': acc_vh
    })
    
    # 2. Voting Ensemble (Soft)
    logger.log("\n--- Soft Voting Ensemble ---")
    voting_soft = create_voting_ensemble(base_models, voting='soft')
    voting_soft_trained, y_pred_vs, cv_vs, cv_std_vs, acc_vs = train_and_evaluate(
        voting_soft, "Voting (Soft)", X_train, X_test, y_train, y_test
    )
    generate_ensemble_confusion_matrix(y_test, y_pred_vs, "Voting_Soft", acc_vs)
    joblib.dump(voting_soft_trained, os.path.join(output_dir, 'voting_soft.pkl'))
    results.append({
        'model': 'Voting (Soft)',
        'cv_accuracy': cv_vs,
        'cv_std': cv_std_vs,
        'test_accuracy': acc_vs
    })
    
    # 3. Stacking Ensemble
    logger.log("\n--- Stacking Ensemble ---")
    stacking = create_stacking_ensemble(base_models)
    stacking_trained, y_pred_st, cv_st, cv_std_st, acc_st = train_and_evaluate(
        stacking, "Stacking", X_train, X_test, y_train, y_test
    )
    generate_ensemble_confusion_matrix(y_test, y_pred_st, "Stacking", acc_st)
    joblib.dump(stacking_trained, os.path.join(output_dir, 'stacking.pkl'))
    results.append({
        'model': 'Stacking',
        'cv_accuracy': cv_st,
        'cv_std': cv_std_st,
        'test_accuracy': acc_st
    })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)
    results_df.to_csv(os.path.join(output_dir, 'ensemble_results.csv'), index=False)
    
    # Generate comparison chart (use 72.21 as baseline from original pipeline)
    generate_ensemble_comparison(results_df, baseline_accuracy=72.21)
    
    # Log summary
    logger.log("\n" + "=" * 60)
    logger.log("ENSEMBLE MODEL RESULTS (sorted by test accuracy)")
    logger.log("=" * 60)
    for _, row in results_df.iterrows():
        logger.log(f"{row['model']:20} | CV: {row['cv_accuracy']:.2f}% (+/-{row['cv_std']:.2f}%) | Test: {row['test_accuracy']:.2f}%")
    logger.log("")
    
    best_model = results_df.iloc[0]
    logger.log(f"🏆 BEST ENSEMBLE: {best_model['model']} with {best_model['test_accuracy']:.2f}% test accuracy")
    
    return results_df


if __name__ == '__main__':
    run()
