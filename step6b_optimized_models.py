"""
WISDM-51 Activity Recognition Pipeline
Step 6b: Optimized Model Training with Hyperparameter Tuning

Uses RandomizedSearchCV with subsampling for fast hyperparameter tuning,
then retrains the best model on full training data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import randint, uniform
import joblib

from config import (DATA_DIR, VIS_DIR, METADATA_COLS, RANDOM_STATE, TEST_SIZE, 
                    CV_FOLDS, N_JOBS, ACTIVITY_NAMES)
from logger import logger

# Subsampling settings for fast hyperparameter tuning
TUNE_SAMPLE_SIZE = 30000  # Use 30K samples for tuning (fast)
N_ITER_SEARCH = 15  # Number of random hyperparameter combinations to try
FAST_CV_FOLDS = 3  # Fewer folds for faster tuning


def get_models_and_params():
    """Return models with parameter distributions for RandomizedSearchCV."""
    
    models_params = {
        'KNN': {
            'model': KNeighborsClassifier(algorithm='ball_tree'),  # Faster than brute force
            'params': {
                'n_neighbors': randint(3, 15),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan'],
                'leaf_size': [20, 30, 40]
            },
            'n_iter': 12
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'params': {
                'max_depth': [10, 15, 20, 25, 30, None],
                'min_samples_split': randint(2, 15),
                'min_samples_leaf': randint(1, 6),
                'criterion': ['gini', 'entropy']
            },
            'n_iter': 15
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
            'params': {
                'n_estimators': [100, 150, 200],
                'max_depth': [15, 20, 25, None],
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 4),
                'max_features': ['sqrt', 'log2']
            },
            'n_iter': 15
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100],  # Reduced from [100, 150, 200]
                'learning_rate': uniform(0.08, 0.07),  # Narrower range: 0.08-0.15
                'max_depth': [3, 5],  # Reduced from [3, 5, 7]
                'subsample': uniform(0.8, 0.2)  # Narrower range: 0.8-1.0
            },
            'n_iter': 8  # Reduced from 15 to 8 iterations
        }
    }
    
    return models_params


def subsample_data(X, y, sample_size, random_state=RANDOM_STATE):
    """Subsample data for faster hyperparameter tuning."""
    if len(X) <= sample_size:
        return X, y
    
    # Stratified subsample
    _, X_sub, _, y_sub = train_test_split(
        X, y, test_size=sample_size, random_state=random_state, stratify=y
    )
    return X_sub, y_sub


def train_optimized_model(X_train, y_train, model_name, model_config):
    """Train model with RandomizedSearchCV on subsampled data, then retrain on full data."""
    logger.log(f"  Optimizing {model_name}...")
    
    # Subsample for fast tuning
    X_tune, y_tune = subsample_data(X_train, y_train, TUNE_SAMPLE_SIZE)
    logger.log(f"    Tuning on {len(X_tune):,} samples...")
    
    cv = StratifiedKFold(n_splits=FAST_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    n_iter = model_config.get('n_iter', N_ITER_SEARCH)
    
    random_search = RandomizedSearchCV(
        model_config['model'],
        model_config['params'],
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    random_search.fit(X_tune, y_tune)
    
    best_params = random_search.best_params_
    logger.log(f"    Best params: {best_params}")
    logger.log(f"    Tuning CV score: {random_search.best_score_*100:.2f}%")
    
    # Retrain on full training data with best params
    logger.log(f"    Retraining on full {len(X_train):,} samples...")
    
    # Clone the model with best parameters
    if model_name == 'KNN':
        final_model = KNeighborsClassifier(**best_params)
    elif model_name == 'DecisionTree':
        final_model = DecisionTreeClassifier(random_state=RANDOM_STATE, **best_params)
    elif model_name == 'RandomForest':
        final_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS, **best_params)
    elif model_name == 'GradientBoosting':
        final_model = GradientBoostingClassifier(random_state=RANDOM_STATE, **best_params)
    else:
        final_model = random_search.best_estimator_
    
    final_model.fit(X_train, y_train)
    
    return final_model, best_params, random_search.best_score_


def generate_optimized_confusion_matrix(y_true, y_pred, model_name, accuracy):
    """Generate confusion matrix for optimized model."""
    vis_dir = os.path.join(VIS_DIR, 'optimized_confusion_matrices')
    os.makedirs(vis_dir, exist_ok=True)
    
    labels = sorted(set(y_true) | set(y_pred))
    label_names = [ACTIVITY_NAMES.get(l, str(l)) for l in labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Optimized {model_name}\nAccuracy: {accuracy:.2f}%')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'optimized_{model_name}_cm.png'), dpi=300)
    plt.close()


def generate_comparison_chart(results_df):
    """Generate comparison chart of optimized models."""
    vis_dir = VIS_DIR
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = results_df['model'].values
    cv_acc = results_df['cv_accuracy'].values
    test_acc = results_df['test_accuracy'].values
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cv_acc, width, label='CV Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, test_acc, width, label='Test Accuracy', color='coral')
    
    for bar, val in zip(bars1, cv_acc):
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    for bar, val in zip(bars2, test_acc):
        ax.annotate(f'{val:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Optimized Model Performance (CV vs Test)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'optimized_model_comparison.png'), dpi=300)
    plt.close()
    
    logger.log("Saved optimized model comparison chart")


def run(selected_df=None):
    """Execute Step 6b: Optimized model training with hyperparameter tuning."""
    logger.header("STEP 6B: Optimized Model Training")
    
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
    
    models_params = get_models_and_params()
    results = []
    
    output_dir = os.path.join(DATA_DIR, '06b_optimized_results')
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, model_config in models_params.items():
        best_model, best_params, cv_score = train_optimized_model(
            X_train, y_train, model_name, model_config
        )
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred) * 100
        
        logger.log(f"    Test accuracy: {test_accuracy:.2f}%")
        
        # Generate confusion matrix
        generate_optimized_confusion_matrix(y_test, y_pred, model_name, test_accuracy)
        
        # Save model
        model_path = os.path.join(output_dir, f'{model_name}_optimized.pkl')
        joblib.dump(best_model, model_path)
        
        results.append({
            'model': model_name,
            'cv_accuracy': cv_score * 100,
            'test_accuracy': test_accuracy,
            'best_params': str(best_params)
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values('test_accuracy', ascending=False)
    results_df.to_csv(os.path.join(output_dir, 'optimized_results.csv'), index=False)
    
    # Generate comparison chart
    generate_comparison_chart(results_df)
    
    # Log summary
    logger.log("\n" + "=" * 60)
    logger.log("OPTIMIZED MODEL RESULTS (sorted by test accuracy)")
    logger.log("=" * 60)
    for _, row in results_df.iterrows():
        logger.log(f"{row['model']:20} | CV: {row['cv_accuracy']:.2f}% | Test: {row['test_accuracy']:.2f}%")
    logger.log("")
    
    return results_df


if __name__ == '__main__':
    run()
