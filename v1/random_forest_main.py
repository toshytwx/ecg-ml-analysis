#!/usr/bin/env python3
"""
Enhanced ECG ML Analysis with Hyperparameter Tuning and Visualization
"""

import os
import sys
import pandas as pd
import time as time_module
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from optimized_feature_extractor import OptimizedFeatureExtractor
import warnings
import pickle

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import ECGDataset

def load_features(input_path='extracted_features.csv'):
    """Load features and labels from CSV file"""
    if not os.path.exists(input_path):
        return None, None
    df = pd.read_csv(input_path)
    y = df['label'].values
    X = df.drop('label', axis=1).values
    print(f"Features loaded from {input_path}")
    return X, y

def create_visualizations(results_df, feature_importance, model_name, output_dir='./output'):
    """Create comprehensive visualizations for model analysis"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Model Performance Comparison
    ax1 = plt.subplot(3, 3, 1)
    metrics = ['Train_Accuracy', 'Test_Accuracy', 'CV_Accuracy']
    values = [results_df[metric].iloc[0] for metric in metrics]
    bars = ax1.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title(f'{model_name} Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Feature Importance
    ax2 = plt.subplot(3, 3, 2)
    top_features = feature_importance.head(10)
    bars = ax2.barh(range(len(top_features)), top_features.values, color='skyblue')
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels([f'Feature {i}' for i in top_features.index])
    ax2.set_xlabel('Importance')
    ax2.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, top_features.values)):
        ax2.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', va='center', fontsize=9)
    
    # 3. Hyperparameter Performance
    ax3 = plt.subplot(3, 3, 3)
    if 'n_estimators' in results_df.columns:
        param_cols = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
        param_cols = [col for col in param_cols if col in results_df.columns]
        
        if len(param_cols) > 0:
            # Create a simple parameter comparison
            param_values = results_df[param_cols].iloc[0]
            param_names = [col.replace('_', ' ').title() for col in param_cols]
            
            # Normalize values for comparison
            normalized_values = []
            for col in param_cols:
                val = param_values[col]
                if col == 'n_estimators':
                    normalized_values.append(val / 200)  # Normalize to 0-1
                elif col == 'max_depth':
                    normalized_values.append(val / 20 if val else 0.5)
                elif col in ['min_samples_split', 'min_samples_leaf']:
                    normalized_values.append(val / 10 if val else 0.1)
                else:
                    normalized_values.append(val)
            
            bars = ax3.bar(param_names, normalized_values, color='lightcoral')
            ax3.set_title('Hyperparameter Values (Normalized)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Normalized Value')
            ax3.tick_params(axis='x', rotation=45)
    
    # 4. Cross-Validation Scores Distribution
    ax4 = plt.subplot(3, 3, 4)
    cv_scores = results_df['CV_Scores'].iloc[0] if 'CV_Scores' in results_df.columns else [results_df['CV_Accuracy'].iloc[0]]
    ax4.hist(cv_scores, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.axvline(np.mean(cv_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cv_scores):.3f}')
    ax4.set_xlabel('CV Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Cross-Validation Scores Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    
    # 5. Model Performance Over Time (if multiple runs)
    ax5 = plt.subplot(3, 3, 5)
    if len(results_df) > 1:
        ax5.plot(range(len(results_df)), results_df['Test_Accuracy'], 'o-', label='Test Accuracy', linewidth=2)
        ax5.plot(range(len(results_df)), results_df['CV_Accuracy'], 's-', label='CV Accuracy', linewidth=2)
        ax5.set_xlabel('Model Run')
        ax5.set_ylabel('Accuracy')
        ax5.set_title('Performance Over Different Runs', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Single Model Run\nNo comparison available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Performance Over Time', fontsize=14, fontweight='bold')
    
    # 6. Feature Importance vs Performance
    ax6 = plt.subplot(3, 3, 6)
    if len(results_df) > 1 and 'Feature_Importance_Sum' in results_df.columns:
        ax6.scatter(results_df['Feature_Importance_Sum'], results_df['Test_Accuracy'], 
                   s=100, alpha=0.7, c=results_df['Test_Accuracy'], cmap='viridis')
        ax6.set_xlabel('Sum of Top 5 Feature Importance')
        ax6.set_ylabel('Test Accuracy')
        ax6.set_title('Feature Importance vs Performance', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Feature Importance\nvs Performance\n(Requires multiple runs)', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Feature Importance vs Performance', fontsize=14, fontweight='bold')
    
    # 7. Model Complexity Analysis
    ax7 = plt.subplot(3, 3, 7)
    if 'n_estimators' in results_df.columns and 'max_depth' in results_df.columns:
        complexity = results_df['n_estimators'].iloc[0] * (results_df['max_depth'].iloc[0] or 10)
        ax7.bar(['Model Complexity'], [complexity], color='orange')
        ax7.set_ylabel('Complexity Score')
        ax7.set_title('Model Complexity', fontsize=14, fontweight='bold')
        ax7.text(0, complexity + complexity*0.05, f'{complexity:.0f}', ha='center', fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'Model Complexity\nAnalysis\n(Parameters not available)', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=12)
        ax7.set_title('Model Complexity', fontsize=14, fontweight='bold')
    
    # 8. Performance Summary Table
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Value'],
        ['Train Accuracy', f"{results_df['Train_Accuracy'].iloc[0]:.3f}"],
        ['Test Accuracy', f"{results_df['Test_Accuracy'].iloc[0]:.3f}"],
        ['CV Accuracy', f"{results_df['CV_Accuracy'].iloc[0]:.3f}"],
        ['CV Std', f"{results_df['CV_Std'].iloc[0]:.3f}"],
        ['Training Time', f"{results_df['Training_Time'].iloc[0]:.2f}s"],
    ]
    
    if 'n_estimators' in results_df.columns:
        summary_data.append(['N Estimators', f"{results_df['n_estimators'].iloc[0]}"])
    if 'max_depth' in results_df.columns:
        summary_data.append(['Max Depth', f"{results_df['max_depth'].iloc[0] or 'None'}"])
    
    table = ax8.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data)):
        for j in range(len(summary_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax8.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    # 9. Feature Importance Heatmap (if multiple features)
    ax9 = plt.subplot(3, 3, 9)
    if len(feature_importance) > 5:
        # Create a heatmap of top features
        top_10_features = feature_importance.head(10)
        feature_matrix = np.array(top_10_features.values).reshape(1, -1)
        
        im = ax9.imshow(feature_matrix, cmap='YlOrRd', aspect='auto')
        ax9.set_xticks(range(len(top_10_features)))
        ax9.set_xticklabels([f'F{i}' for i in top_10_features.index], rotation=45)
        ax9.set_yticks([0])
        ax9.set_yticklabels(['Importance'])
        ax9.set_title('Top 10 Features Heatmap', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax9, shrink=0.8)
        cbar.set_label('Importance Score')
    else:
        ax9.text(0.5, 0.5, 'Feature Heatmap\n(Insufficient features)', 
                ha='center', va='center', transform=ax9.transAxes, fontsize=12)
        ax9.set_title('Feature Importance Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()

def run_hyperparameter_tuning(X, y, cv_folds=5):
    """Run hyperparameter tuning for Random Forest"""
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    print(f"Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])} parameter combinations...")
    
    # Create base model
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X, y)
    
    print(f"\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

def main():
    """Main function for enhanced ECG ML analysis"""
    
    print("="*80)
    print("ENHANCED ECG ML ANALYSIS - HYPERPARAMETER TUNING & VISUALIZATION")
    print("="*80)
    
    # Configuration
    dataset_path = "/Users/dmytro/Diploma/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0"
    output_dir = "./output"
    batch_size = 50
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load dataset and extract features
    features_path = 'extracted_features.csv'
    
    # Try to load existing features first
    X, y = load_features(features_path)
    if X is not None and y is not None:
        print(f"Loaded existing features:")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Number of age groups: {len(np.unique(y))}")
    else:
        print("\n1. Loading dataset and extracting features...")
        dataset = ECGDataset(dataset_path)
        feature_extractor = OptimizedFeatureExtractor()
        
        valid_records = dataset.get_valid_records()
        total_records = len(valid_records)
        print(f"Processing {total_records} valid records...")
        
        features_list = []
        labels_list = []
        
        # Process in batches
        num_batches = (total_records + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_records)
            batch_records = valid_records[start_idx:end_idx]
            
            print(f"  Processing batch {batch_idx + 1}/{num_batches} ({len(batch_records)} records)...")
            
            batch_start_time = time_module.time()  # Use the renamed module
            
            for record_id in batch_records:
                record = dataset.load_record(record_id)
                if record is None:
                    continue
                
                features = feature_extractor.extract_features(record)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(record['subject_info']['Age_group'])
            
            batch_time = time_module.time() - batch_start_time
            print(f"    ✓ Batch completed in {batch_time:.2f}s ({len(batch_records)/batch_time:.1f} records/sec)")
        
        if len(features_list) == 0:
            print("No features extracted!")
            return
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"\nFeature extraction completed!")
        print(f"  Total records processed: {len(features_list)}")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Number of age groups: {len(np.unique(y))}")
    
    # 2. Split data
    print("\n2. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 3. Hyperparameter tuning
    print("\n3. Running hyperparameter tuning...")
    best_model, best_params, cv_results = run_hyperparameter_tuning(X_train, y_train)
    
    # 4. Train best model
    print("\n4. Training best model...")
    start_time = time_module.time()
    
    best_model.fit(X_train, y_train)
    training_time = time_module.time() - start_time
    
    # 5. Evaluate model
    print("\n5. Evaluating model...")
    
    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    # Scores
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    # 6. Feature importance
    feature_importance = pd.Series(best_model.feature_importances_, 
                                 index=range(X.shape[1])).sort_values(ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    for i, (feature_idx, importance) in enumerate(feature_importance.head(10).items()):
        print(f"  {i+1:2d}. Feature {feature_idx:2d}: {importance:.4f}")
    
    # 7. Create results dataframe
    results_df = pd.DataFrame({
        'Model': ['RandomForest_Tuned'],
        'Train_Accuracy': [train_accuracy],
        'Test_Accuracy': [test_accuracy],
        'CV_Accuracy': [cv_mean],
        'CV_Std': [cv_std],
        'Training_Time': [training_time],
        'CV_Scores': [cv_scores.tolist()],
        'n_estimators': [best_params.get('n_estimators', None)],
        'max_depth': [best_params.get('max_depth', None)],
        'min_samples_split': [best_params.get('min_samples_split', None)],
        'min_samples_leaf': [best_params.get('min_samples_leaf', None)],
        'max_features': [best_params.get('max_features', None)],
        'Feature_Importance_Sum': [feature_importance.head(5).sum()]
    })
    
    # 8. Save results
    results_path = os.path.join(output_dir, 'random_forest_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # 9. Create visualizations
    print("\n6. Creating visualizations...")
    create_visualizations(results_df, feature_importance, "Random Forest Tuned", output_dir)
    
    # 10. Save model
    model_path = os.path.join(output_dir, 'random_forest_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Model saved to: {model_path}")
    
    # 11. Classification report
    print("\n7. Detailed Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    print("\n" + "="*80)
    print("ENHANCED ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
