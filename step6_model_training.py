"""
WISDM-51 Activity Recognition Pipeline
Step 6: Model Training and Evaluation

Trains KNN, Naive Bayes, Decision Tree, and Random Forest on each scaled dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import (DATA_DIR, VIS_DIR, SCALER_NAMES, METADATA_COLS, RANDOM_STATE, 
                    TEST_SIZE, ACTIVITY_NAMES)
from logger import logger


def get_models():
    """Return dictionary of model instances."""
    return {
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'NaiveBayes': GaussianNB(),
        'DecisionTree': DecisionTreeClassifier(max_depth=20, random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=20, 
                                                random_state=RANDOM_STATE, n_jobs=-1)
    }


def generate_confusion_matrix(y_true, y_pred, scaler_name, model_name, accuracy):
    """Generate and save confusion matrix visualization."""
    vis_dir = os.path.join(VIS_DIR, 'confusion_matrices')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get unique labels
    labels = sorted(set(y_true) | set(y_pred))
    label_names = [ACTIVITY_NAMES.get(l, l) for l in labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{scaler_name.upper()} + {model_name}\nAccuracy: {accuracy:.2f}%')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'{scaler_name}_{model_name}_cm.png'), dpi=300)
    plt.close()


def train_and_evaluate(X_train, X_test, y_train, y_test, scaler_name, model_name, model):
    """Train model and return evaluation results."""
    logger.log(f"  Training {model_name}...")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred) * 100
    logger.log(f"    Accuracy: {accuracy:.2f}%")
    
    # Generate confusion matrix
    generate_confusion_matrix(y_test, y_pred, scaler_name, model_name, accuracy)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        'scaler': scaler_name,
        'model': model_name,
        'accuracy': accuracy,
        'macro_precision': report['macro avg']['precision'] * 100,
        'macro_recall': report['macro avg']['recall'] * 100,
        'macro_f1': report['macro avg']['f1-score'] * 100,
        'weighted_f1': report['weighted avg']['f1-score'] * 100
    }


def run(selected_dfs=None):
    """Execute Step 6: Train and evaluate all model/scaler combinations."""
    logger.header("STEP 6: Model Training and Evaluation")
    
    output_dir = os.path.join(DATA_DIR, '06_results')
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for scaler_name in SCALER_NAMES:
        logger.log(f"\nEvaluating {scaler_name} scaled data...")
        
        if selected_dfs and scaler_name in selected_dfs:
            df = selected_dfs[scaler_name]
        else:
            df = pd.read_csv(os.path.join(DATA_DIR, '05_selected', f'{scaler_name}_selected.csv'))
        
        feature_cols = [c for c in df.columns if c not in METADATA_COLS]
        
        X = df[feature_cols].values
        y = df['activity_label'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        logger.log(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        models = get_models()
        for model_name, model in models.items():
            result = train_and_evaluate(X_train, X_test, y_train, y_test, 
                                        scaler_name, model_name, model)
            all_results.append(result)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    results_df.to_csv(os.path.join(output_dir, 'model_results.csv'), index=False)
    
    logger.log("\n" + "="*60)
    logger.log("MODEL RESULTS (sorted by accuracy)")
    logger.log("="*60)
    
    for _, row in results_df.iterrows():
        logger.log(f"{row['scaler']:8} + {row['model']:12} = {row['accuracy']:.2f}%")
    
    logger.log("")
    return results_df


if __name__ == '__main__':
    run()
