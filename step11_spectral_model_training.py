"""
WISDM-51 Activity Recognition Pipeline
Step 11: Spectral Feature Model Training and Evaluation

Trains KNN, Naive Bayes, Decision Tree, and Random Forest on spectral features.
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
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)

from config import DATA_DIR, VIS_DIR, METADATA_COLS, RANDOM_STATE, TEST_SIZE, ACTIVITY_NAMES
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


def generate_confusion_matrix(y_true, y_pred, model_name, accuracy):
    """Generate and save confusion matrix visualization."""
    vis_dir = os.path.join(VIS_DIR, 'spectral_confusion_matrices')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get unique labels
    labels = sorted(set(y_true) | set(y_pred))
    label_names = [ACTIVITY_NAMES.get(l, str(l)) for l in labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Spectral Features - {model_name}\nAccuracy: {accuracy:.2f}%')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'spectral_{model_name}_cm.png'), dpi=300)
    plt.close()


def generate_model_comparison_chart(results_df):
    """Generate bar chart comparing all models."""
    vis_dir = VIS_DIR
    os.makedirs(vis_dir, exist_ok=True)
    
    models = results_df['model'].values
    metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = results_df[metric].values
        bars = ax.bar(x + i * width, values, width, label=label)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score (%)')
    ax.set_title('Spectral Features - Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'spectral_model_comparison.png'), dpi=300)
    plt.close()
    
    logger.log("Saved model comparison chart")


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model):
    """Train model and return evaluation results."""
    logger.log(f"  Training {model_name}...")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
    macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100
    weighted_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    weighted_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    
    logger.log(f"    Accuracy: {accuracy:.2f}%")
    
    # Generate confusion matrix
    generate_confusion_matrix(y_test, y_pred, model_name, accuracy)
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }


def run(selected_df=None):
    """Execute Step 11: Train and evaluate models on spectral features."""
    logger.header("STEP 11: Spectral Model Training and Evaluation")
    
    # Load selected spectral features if not provided
    if selected_df is None:
        input_path = os.path.join(DATA_DIR, '10_spectral_selected', 'FINAL_SELECTED_SPECTRAL_FEATURES.csv')
        logger.log(f"Loading selected features from {input_path}")
        selected_df = pd.read_csv(input_path)
    
    logger.log(f"Training on {len(selected_df):,} samples...")
    
    # Get feature columns
    feature_cols = [c for c in selected_df.columns if c not in METADATA_COLS]
    logger.log(f"Number of features: {len(feature_cols)}")
    
    # Prepare data
    X = selected_df[feature_cols].values
    y = selected_df['activity_label'].values
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.log(f"Train set: {len(X_train):,} samples")
    logger.log(f"Test set: {len(X_test):,} samples")
    
    # Train and evaluate each model
    all_results = []
    models = get_models()
    
    for model_name, model in models.items():
        result = train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model)
        all_results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    # Save results
    output_dir = os.path.join(DATA_DIR, '11_spectral_results')
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, 'spectral_model_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.log(f"\nSaved results to {results_path}")
    
    # Generate comparison chart
    generate_model_comparison_chart(results_df)
    
    # Log summary
    logger.log("\n" + "=" * 60)
    logger.log("SPECTRAL MODEL RESULTS (sorted by accuracy)")
    logger.log("=" * 60)
    
    for _, row in results_df.iterrows():
        logger.log(f"{row['model']:12} | Acc: {row['accuracy']:.2f}% | "
                  f"Prec: {row['macro_precision']:.2f}% | "
                  f"Rec: {row['macro_recall']:.2f}% | "
                  f"F1: {row['macro_f1']:.2f}%")
    
    logger.log("")
    
    return results_df


if __name__ == '__main__':
    run()
