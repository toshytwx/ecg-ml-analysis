#!/usr/bin/env python3
"""
ECG Classification using Autoencoder Features
============================================

This script uses the trained autoencoder as a feature extractor
and trains a classifier on the learned representations.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import wfdb

# Import the autoencoder model
from train_autoencoder import ECGAutoencoder, LATENT_DIM, ENCODER_CHANNELS, DECODER_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, SIGNAL_LENGTH

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data configuration
DATA_DIR = "input/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0"
AUTOENCODER_PATH = "ecg_autoencoder_outputs/best_autoencoder.pth"
OUTPUT_DIR = "ecg_autoencoder_outputs"
SIGNAL_LENGTH = 2500
SIGNAL_DOWNSAMPLE = 1
BATCH_SIZE = 64
NUM_WORKERS = 0

# Classification configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Device configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_ecg_data():
    """Load and preprocess ECG data"""
    print("🔄 Loading ECG data...")
    
    # Load subject information from CSV (same as RNN script)
    subject_info_csv = os.path.join(DATA_DIR, "subject-info.csv")
    if not os.path.exists(subject_info_csv):
        print(f"❌ Subject info CSV not found: {subject_info_csv}")
        sys.exit(1)
    
    subject_info = pd.read_csv(subject_info_csv, index_col=0)
    print(f"Loaded subject info: {len(subject_info)} subjects")
    print(f"Age groups available: {subject_info['Age_group'].unique()}")
    
    # Get all .dat files
    dat_files = list(Path(DATA_DIR).glob("*.dat"))
    print(f"Found {len(dat_files)} ECG files")
    
    signals = []
    ages = []
    
    for i, dat_file in enumerate(tqdm(dat_files, desc="Loading signals")):
        try:
            # Read signal
            record_name = dat_file.stem
            signal, fields = wfdb.rdsamp(str(dat_file.parent / record_name))
            
            # Use first lead (Lead I)
            if signal.ndim > 1:
                signal = signal[:, 0]
            
            # Downsample if needed
            if SIGNAL_DOWNSAMPLE > 1:
                signal = signal[::SIGNAL_DOWNSAMPLE]
            
            # Pad or truncate to fixed length
            if len(signal) > SIGNAL_LENGTH:
                signal = signal[:SIGNAL_LENGTH]
            else:
                signal = np.pad(signal, (0, SIGNAL_LENGTH - len(signal)), 'constant')
            
            # Normalize signal
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            signal = signal.astype(np.float32)
            
            signals.append(signal)
            
            # Get age from subject info CSV
            try:
                record_id = int(record_name)
                if record_id in subject_info.index:
                    age_group = subject_info.loc[record_id, 'Age_group']
                    ages.append(age_group)
                else:
                    print(f"Warning: Record {record_id} not found in subject info")
                    ages.append(3)  # Default age group
            except:
                ages.append(3)  # Default age group if parsing fails
                
        except Exception as e:
            print(f"Error loading {dat_file}: {e}")
            continue
    
    signals = np.array(signals)
    ages = np.array(ages)
    
    # Reshape signals to have channel dimension
    signals = signals.reshape(signals.shape[0], 1, signals.shape[1])  # (N, 1, 2500)
    
    print(f"Loaded {len(signals)} signals")
    print(f"Signal shape: {signals.shape}")
    print(f"Age groups: {np.unique(ages)}")
    
    # Convert ages to integers and handle NaN values
    ages_clean = []
    for age in ages:
        if pd.isna(age):
            ages_clean.append(3)  # Default age group for NaN
        else:
            ages_clean.append(int(age))
    
    ages_clean = np.array(ages_clean)
    print(f"Age distribution: {np.bincount(ages_clean)}")
    
    return signals, ages_clean

def create_age_groups(ages):
    """Create age groups for classification - use the age groups directly"""
    # The ages are already age groups (1-15), so use them directly
    # Just convert to 0-based indexing for sklearn
    return np.array(ages) - 1

# =============================================================================
# AUTOENCODER FEATURE EXTRACTION
# =============================================================================

class AutoencoderFeatureExtractor:
    """Feature extractor using trained autoencoder"""
    
    def __init__(self, model_path, device):
        self.device = device
        self.model = self.load_autoencoder(model_path)
        self.model.eval()
        
    def load_autoencoder(self, model_path):
        """Load trained autoencoder"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Autoencoder model not found: {model_path}")
        
        # Create model
        model = ECGAutoencoder(
            input_length=SIGNAL_LENGTH,
            latent_dim=LATENT_DIM,
            encoder_channels=ENCODER_CHANNELS,
            decoder_channels=DECODER_CHANNELS,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING
        ).to(self.device)
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def extract_features(self, signals):
        """Extract features from ECG signals using autoencoder"""
        features = []
        
        # Create data loader
        dataset = TensorDataset(torch.FloatTensor(signals))
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                batch_signals = batch[0].to(self.device)
                
                # Extract latent representations
                latent_features = self.model.encode(batch_signals)
                features.append(latent_features.cpu().numpy())
        
        return np.vstack(features)

# =============================================================================
# CLASSIFICATION MODELS
# =============================================================================

def train_classifiers(X_train, X_test, y_train, y_test):
    """Train multiple classifiers and compare performance"""
    print("\n🎯 Training Classification Models")
    print("=" * 40)
    
    # Enhanced feature engineering
    print("🔧 Applying feature engineering...")
    
    # 1. Handle class imbalance with SMOTE
    from imblearn.over_sampling import SMOTE
    print("  🔄 Applying SMOTE for class balance...")
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"  ✅ Balanced training samples: {len(X_train_balanced)} (was {len(X_train)})")
    
    # 2. Polynomial features for non-linear relationships (reduced complexity)
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_balanced)
    X_test_poly = poly.transform(X_test)
    
    # 3. Feature selection to reduce overfitting (fewer features for speed)
    from sklearn.feature_selection import SelectKBest, f_classif
    k_best = SelectKBest(score_func=f_classif, k=min(50, X_train_poly.shape[1]))  # Reduced from 100 to 50
    X_train_selected = k_best.fit_transform(X_train_poly, y_train_balanced)
    X_test_selected = k_best.transform(X_test_poly)
    
    print(f"  ✅ Original features: {X_train.shape[1]}")
    print(f"  ✅ Polynomial features: {X_train_poly.shape[1]}")
    print(f"  ✅ Selected features: {X_train_selected.shape[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=20, 
            min_samples_split=5, 
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'Logistic Regression': LogisticRegression(
            C=0.1, 
            penalty='l2', 
            solver='liblinear',
            class_weight='balanced',
            random_state=RANDOM_STATE, 
            max_iter=2000
        ),
        'SVM': SVC(
            C=1.0, 
            kernel='rbf', 
            gamma='scale',
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=50,  # Reduced from 200
            learning_rate=0.2,  # Increased to compensate
            max_depth=4,  # Reduced from 6
            min_samples_split=20,  # Increased to reduce complexity
            min_samples_leaf=10,  # Increased to reduce complexity
            subsample=0.8,  # Add subsampling for speed
            random_state=RANDOM_STATE
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=50,  # Reduced from 200
            learning_rate=0.2,  # Increased to compensate
            random_state=RANDOM_STATE
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='minkowski'
        ),
        'Naive Bayes': GaussianNB(),
        'Voting Ensemble': VotingClassifier([
            ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE))
        ], voting='soft'),
        'Stacking Ensemble': StackingClassifier([
            ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE))
        ], final_estimator=LogisticRegression(random_state=RANDOM_STATE))
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\n📊 Training {name}...")
        
        # Hyperparameter tuning for top models (simplified for speed)
        if name in ['Random Forest']:  # Only tune Random Forest, skip others for speed
            print(f"  🔧 Tuning hyperparameters for {name}...")
            
            param_grid = {
                'n_estimators': [100, 200],  # Reduced options
                'max_depth': [10, 15],  # Reduced options
                'min_samples_split': [5, 10],  # Reduced options
                'min_samples_leaf': [2, 4]  # Reduced options
            }
            
            # Use GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                clf, param_grid, cv=2, scoring='accuracy',  # Reduced CV folds
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train_scaled, y_train_balanced)
            clf = grid_search.best_estimator_
            print(f"  ✅ Best params: {grid_search.best_params_}")
        
        # Train classifier
        start_time = time.time()
        clf.fit(X_train_scaled, y_train_balanced)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = clf.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_train_scaled, y_train_balanced, cv=CV_FOLDS, scoring='accuracy')
        
        results[name] = {
            'classifier': clf,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'predictions': y_pred
        }
        
        print(f"  ✅ Accuracy: {accuracy:.4f}")
        print(f"  ✅ F1-Macro: {f1_macro:.4f}")
        print(f"  ✅ F1-Weighted: {f1_weighted:.4f}")
        print(f"  ✅ CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  ✅ Training Time: {training_time:.2f}s")
    
    return results, scaler

# =============================================================================
# VISUALIZATION AND ANALYSIS
# =============================================================================

def generate_classification_analysis(results, y_test, age_groups):
    """Generate comprehensive classification analysis"""
    print("\n📊 Generating Classification Analysis...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    f1_macros = [results[model]['f1_macro'] for model in models]
    f1_weighteds = [results[model]['f1_weighted'] for model in models]
    cv_means = [results[model]['cv_mean'] for model in models]
    cv_stds = [results[model]['cv_std'] for model in models]
    
    # Accuracy bar plot
    bars1 = axes[0, 0].bar(models, accuracies, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Classification Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # F1-Score comparison
    x = np.arange(len(models))
    width = 0.25
    
    bars2 = axes[0, 1].bar(x - width, f1_macros, width, label='F1-Macro', alpha=0.7, color='lightcoral')
    bars3 = axes[0, 1].bar(x, f1_weighteds, width, label='F1-Weighted', alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)
    
    # Cross-validation scores
    bars4 = axes[1, 0].bar(models, cv_means, alpha=0.7, color='gold', yerr=cv_stds, capsize=5)
    axes[1, 0].set_title('Cross-Validation Scores', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('CV Accuracy')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Training time comparison
    training_times = [results[model]['training_time'] for model in models]
    bars5 = axes[1, 1].bar(models, training_times, alpha=0.7, color='plum')
    axes[1, 1].set_title('Training Time', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/autoencoder_classification_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion matrices for each model
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    age_group_names = [f'Age_{i+1}' for i in range(15)]
    
    for i, (model_name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=age_group_names, yticklabels=age_group_names)
        axes[i].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.3f}', fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/autoencoder_classification_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Detailed classification report
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_result = results[best_model]
    
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    print(f"   F1-Macro: {best_result['f1_macro']:.4f}")
    print(f"   F1-Weighted: {best_result['f1_weighted']:.4f}")
    
    print(f"\n📊 Detailed Classification Report ({best_model}):")
    print(classification_report(y_test, best_result['predictions'], target_names=age_group_names))
    
    return best_model, best_result

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main classification pipeline"""
    print("🏥 ECG Classification using Autoencoder Features")
    print("=" * 50)
    
    # Check if autoencoder exists
    if not os.path.exists(AUTOENCODER_PATH):
        print(f"❌ Autoencoder model not found: {AUTOENCODER_PATH}")
        print("Please train the autoencoder first using train_autoencoder.py")
        sys.exit(1)
    
    # Load data
    signals, ages = load_ecg_data()
    age_groups = create_age_groups(ages)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  • Total samples: {len(signals)}")
    print(f"  • Age groups: {len(np.unique(age_groups))}")
    print(f"  • Class distribution: {np.bincount(age_groups)}")
    
    # Extract features using autoencoder
    print(f"\n🔍 Extracting features using autoencoder...")
    feature_extractor = AutoencoderFeatureExtractor(AUTOENCODER_PATH, DEVICE)
    features = feature_extractor.extract_features(signals)
    
    print(f"  ✅ Extracted features shape: {features.shape}")
    print(f"  ✅ Feature dimension: {features.shape[1]} (latent space)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, age_groups, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=age_groups
    )
    
    print(f"\n📊 Train/Test Split:")
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Test samples: {len(X_test)}")
    print(f"  • Feature dimension: {X_train.shape[1]}")
    
    # Train classifiers
    results, scaler = train_classifiers(X_train, X_test, y_train, y_test)
    
    # Generate analysis
    best_model, best_result = generate_classification_analysis(results, y_test, age_groups)
    
    # Save results
    results_summary = {
        'best_model': best_model,
        'best_accuracy': best_result['accuracy'],
        'best_f1_macro': best_result['f1_macro'],
        'best_f1_weighted': best_result['f1_weighted'],
        'feature_dimension': features.shape[1],
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    # Save to file
    import json
    with open(f"{OUTPUT_DIR}/autoencoder_classification_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n🎉 Classification completed!")
    print(f"📁 Results saved to: {OUTPUT_DIR}/")
    print(f"📊 Generated files:")
    print(f"  • autoencoder_classification_performance.png")
    print(f"  • autoencoder_classification_confusion_matrices.png")
    print(f"  • autoencoder_classification_results.json")
    
    return results, best_model, best_result

if __name__ == "__main__":
    main()
