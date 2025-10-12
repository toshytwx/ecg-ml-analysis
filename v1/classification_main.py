#!/usr/bin/env python3
"""
Enhanced Linear Classification Analysis with Hyperparameter Tuning
"""

import os
import sys
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
from collections import Counter
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import ECGDataset
from feature_extractor import ECGFeatureExtractor


def load_features(input_path='extracted_features.csv'):
    """Load features and labels from CSV file"""
    if not os.path.exists(input_path):
        return None, None
    df = pd.read_csv(input_path)
    y = df['label'].values
    X = df.drop('label', axis=1).values
    print(f"Features loaded from {input_path}")
    return X, y

def run_hyperparameter_tuning(X_train, y_train):
    """Run hyperparameter tuning for linear models"""
    
    print("\nHyperparameter Tuning")
    print("="*60)
    
    # Define parameter grids
    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'max_iter': [10000],
        'solver': ['lbfgs', 'newton-cg', 'sag'],
    }
    
    svm_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'max_iter': [20000],
        'multi_class': ['ovr'],
        'dual': [False]
    }
    
    results = {}
    
    # Tune LogisticRegression
    print("\nTuning Logistic Regression...")
    lr_grid = GridSearchCV(
        LogisticRegression(random_state=42, class_weight='balanced'),
        lr_param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1,
    )
    lr_grid.fit(X_train, y_train)
    results['LogisticRegression'] = {
        'best_params': lr_grid.best_params_,
        'best_score': lr_grid.best_score_,
        'best_estimator': lr_grid.best_estimator_,
        'cv_results': lr_grid.cv_results_
    }
    
    # Tune LinearSVC
    print("\nTuning Linear SVM...")
    svm_grid = GridSearchCV(
        LinearSVC(random_state=42, class_weight='balanced'),
        svm_param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    svm_grid.fit(X_train, y_train)
    results['LinearSVM'] = {
        'best_params': svm_grid.best_params_,
        'best_score': svm_grid.best_score_,
        'best_estimator': svm_grid.best_estimator_,
        'cv_results': svm_grid.cv_results_
    }
    
    return results

def create_enhanced_visualizations(results, output_dir='./output'):
    """Create comprehensive visualizations for model analysis"""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a 3x3 subplot grid
    fig = plt.figure(figsize=(20, 20))
    
    # 1. Model Comparison
    ax1 = plt.subplot(331)
    models = list(results.keys())
    metrics = ['train_accuracy', 'test_accuracy', 'val_accuracy']
    values = {model: [results[model][m] for m in metrics] for model in models}
    
    x = np.arange(len(metrics))
    width = 0.35
    rects1 = ax1.bar(x - width/2, values[models[0]], width, label=models[0])
    rects2 = ax1.bar(x + width/2, values[models[1]], width, label=models[1])
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    
    # 2. Hyperparameter Impact (LogisticRegression)
    ax2 = plt.subplot(332)
    cv_results = results['LogisticRegression']['cv_results']
    C_values = cv_results['param_C']
    mean_scores = cv_results['mean_test_score']
    ax2.semilogx(C_values, mean_scores, 'o-')
    ax2.set_xlabel('C (regularization parameter)')
    ax2.set_ylabel('Mean CV Score')
    ax2.set_title('LogisticRegression: C Parameter Impact')
    ax2.grid(True)
    
    # 3. Training Time Comparison
    ax3 = plt.subplot(333)
    times = [results[model]['training_time'] for model in models]
    ax3.bar(models, times)
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Training Time Comparison')
    
    # 4. Confusion Matrices
    ax4 = plt.subplot(334)
    cm = results['LogisticRegression']['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', ax=ax4)
    ax4.set_title('LogisticRegression Confusion Matrix')
    
    ax5 = plt.subplot(335)
    cm = results['LinearSVM']['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', ax=ax5)
    ax5.set_title('LinearSVM Confusion Matrix')
    
    # 5. Feature Importance (LogisticRegression coefficients)
    ax6 = plt.subplot(336)
    coef = results['LogisticRegression']['best_estimator'].coef_[0]
    top_n = 10
    indices = np.argsort(np.abs(coef))[-top_n:]
    ax6.barh(range(top_n), coef[indices])
    ax6.set_yticks(range(top_n))
    ax6.set_yticklabels([f'Feature {i}' for i in indices])
    ax6.set_title('Top Feature Coefficients')
    
    # 6. Cross-validation Score Distribution
    ax7 = plt.subplot(337)
    cv_scores = cross_val_score(
        results['LogisticRegression']['best_estimator'],
        results['LogisticRegression']['X_val'],
        results['LogisticRegression']['y_val'],
        cv=5
    )
    ax7.hist(cv_scores, bins=10)
    ax7.axvline(np.mean(cv_scores), color='r', linestyle='--')
    ax7.set_title('Cross-validation Score Distribution')
    
    # 7. Class Distribution
    ax8 = plt.subplot(338)
    class_dist = Counter(results['LogisticRegression']['y_true'])
    ax8.bar(list(class_dist.keys()), list(class_dist.values()))
    ax8.set_title('Class Distribution')
    ax8.set_xlabel('Class')
    ax8.set_ylabel('Count')
    
    # 8. Performance Summary Table
    ax9 = plt.subplot(339)
    ax9.axis('off')
    
    summary_data = []
    for model in models:
        summary_data.extend([
            [f"{model} Metrics", "Value"],
            ["Train Accuracy", f"{results[model]['train_accuracy']:.3f}"],
            ["Test Accuracy", f"{results[model]['test_accuracy']:.3f}"],
            ["Val Accuracy", f"{results[model]['val_accuracy']:.3f}"],
            ["Best C", f"{results[model]['best_params']['C']}"],
            ["Training Time", f"{results[model]['training_time']:.2f}s"],
            ["", ""]
        ])
    
    table = ax9.table(cellText=summary_data,
                     loc='center',
                     cellLoc='left',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax9.set_title('Performance Summary', pad=20)
    
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(output_dir, 'linear_models_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.show()

def main():
    """Main function for enhanced linear classification analysis"""
    
    print("="*80)
    print("LINEAR CLASSIFICATION ANALYSIS")
    print("="*80)
    
    # Configuration
    dataset_path = "/Users/dmytro/Diploma/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load dataset and extract features
    features_path = 'extracted_features.csv'
    X, y = load_features(features_path)
    if X is None or y is None:
        print("No existing features found. Extracting new features...")
        dataset = ECGDataset(dataset_path)
        feature_extractor = ECGFeatureExtractor()
        valid_records = dataset.get_valid_records()
        features_list, labels_list = [], []
        for record_id in valid_records:
            record = dataset.load_record(record_id)
            if record is None:
                continue
            features = feature_extractor.extract_features(record)
            if features is not None:
                features_list.append(features)
                labels_list.append(record['subject_info']['Age_group'])
        X = np.array(features_list)
        y = np.array(labels_list) - 1  # zero-based
    
    print("before Imputer and Scaler")
    print("Nans: " + str(np.any(np.isnan(X))) + " Infs: " + str(np.any(np.isinf(X))))
    print(f"Feature matrix shape: {X.shape}, Labels shape: {y.shape}")
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    # Handle missing values first
    imputer = SimpleImputer(strategy="mean")  # or "median"
    X = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print("after Imputer and Scaler")
    print("Nans: " + str(np.any(np.isnan(X))) + " Infs: " + str(np.any(np.isinf(X))))
    print(f"Feature matrix shape: {X.shape}, Labels shape: {y.shape}")
    exit(0)
    
    # 3. Split data
    print("\n3. Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    print(f"Original training set shape: {Counter(y_train)}")
    print(f"Resampled training set shape: {Counter(y_train_res)}")

    # 4. Hyperparameter tuning FIRST
    print("\n4. Running hyperparameter tuning...")
    tuning_start_time = time.time()
    tuning_results = run_hyperparameter_tuning(X_train_res, y_train_res)
    
    # 5. Train & evaluate best models
    results = {}
    for model_name, model_results in tuning_results.items():
        best_model = model_results['best_estimator']
        
        # Train and evaluate best model
        print(f"\nTraining and evaluating best {model_name}...")
        train_start_time = time.time()
        best_model.fit(X_train, y_train)
        training_time = time.time() - train_start_time
        
        # Predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        y_val_pred = best_model.predict(X_val)
        
        # Store results
        results[model_name] = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'training_time': training_time,
            'best_params': model_results['best_params'],
            'best_score': model_results['best_score'],
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'best_estimator': best_model,
            'cv_results': model_results['cv_results'],
            'X_val': X_val,
            'y_val': y_val,
            'y_true': y_test,
            'y_pred': y_test_pred
        }
    
    # 6. Save results
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Train_Accuracy': [r['train_accuracy'] for r in results.values()],
        'Test_Accuracy': [r['test_accuracy'] for r in results.values()],
        'Val_Accuracy': [r['val_accuracy'] for r in results.values()],
        'Training_Time': [r['training_time'] for r in results.values()]
    })
    
    results_path = os.path.join(output_dir, 'linear_classification_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # 7. Create visualizations
    print("\n7. Creating visualizations...")
    create_enhanced_visualizations(results, output_dir)
    
    print("\n" + "="*80)
    print("LINEAR CLASSIFICATION ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()