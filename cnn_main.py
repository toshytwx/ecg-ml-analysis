#!/usr/bin/env python3
"""
Enhanced ECG ML Analysis using CNN with Hyperparameter Tuning and Visualization
"""

import os
import sys
import pandas as pd
import time as time_module
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from collections import Counter



warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import ECGDataset
from optimized_feature_extractor import OptimizedFeatureExtractor


def load_features(input_path='extracted_features.csv'):
    """Load features and labels from CSV file"""
    if not os.path.exists(input_path):
        return None, None
    df = pd.read_csv(input_path)
    y = df['label'].values
    X = df.drop('label', axis=1).values
    print(f"Features loaded from {input_path}")
    return X, y

def focal_loss(alpha=0.25, gamma=2.0):
    """Focal loss for sparse multi-class labels."""
    import tensorflow as tf
    def loss(y_true, y_pred):
        # y_true: (batch,) sparse ints
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)  # probabilities
        # one-hot encode
        y_true_ohe = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        # cross-entropy
        ce = -y_true_ohe * tf.math.log(y_pred)
        weight = alpha * tf.pow(1.0 - y_pred, gamma)
        fl = weight * ce
        per_sample = tf.reduce_sum(fl, axis=1)
        return tf.reduce_mean(per_sample)
    return loss

def create_cnn_model(input_shape, num_classes, n_conv_layers=2):
    """Create CNN model architecture (now respects n_conv_layers)"""
    print(f"Creating model with input shape {input_shape}, {num_classes} classes, {n_conv_layers} conv layers")
    model = Sequential()
    # add conv blocks dynamically with BatchNorm + L2
    for i in range(n_conv_layers):
        filters = 32 * (2 ** i)
        if i == 0:
            model.add(Conv1D(filters, kernel_size=3, padding='same',
                             activation=None,
                             kernel_regularizer=regularizers.l2(1e-4),
                             input_shape=input_shape))
        else:
            model.add(Conv1D(filters, kernel_size=3, padding='same',
                             activation=None,
                             kernel_regularizer=regularizers.l2(1e-4)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(MaxPooling1D(pool_size=2, padding='same'))
    # use global pooling instead of flatten to reduce params
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def run_hyperparameter_tuning(X_train, y_train, X_val, y_val, class_weights=None):
    """Run hyperparameter tuning for CNN"""
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Define parameter grid
    learning_rates = [0.001, 0.0001]
    batch_sizes = [32, 64]
    conv_layers = [1, 2]
    
    best_val_acc = 0
    best_params = {}
    best_model = None
    
    total_combinations = len(learning_rates) * len(batch_sizes) * len(conv_layers)
    print(f"Testing {total_combinations} parameter combinations...")
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for n_conv_layers in conv_layers:
                print(f"\nTesting: lr={lr}, batch_size={batch_size}, conv_layers={n_conv_layers}")
                
                # Print input shape for debugging
                print(f"Input shape: {(X_train.shape[1], 1)}")
                
                model = create_cnn_model(
                    input_shape=(X_train.shape[1], 1),
                    num_classes=len(np.unique(y_train)),
                    n_conv_layers=n_conv_layers
                )
                
                model.compile(
                   optimizer=Adam(learning_rate=lr),
                   loss=focal_loss(alpha=0.25, gamma=2.0),
                   metrics=['accuracy']
                )
                
                # use provided class_weights if available, else compute
                if class_weights is None:
                    try:
                        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                        cw = {int(cls): float(w) for cls, w in zip(np.unique(y_train), weights)}
                    except Exception:
                        cw = None
                else:
                    cw = class_weights
                
                # callbacks: early stop on val_loss + reduce LR
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=8,
                    restore_best_weights=True,
                    verbose=1
                )
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
                
                history = model.fit(
                    X_train.reshape(-1, X_train.shape[1], 1).astype('float32'),
                    y_train.astype('int32'),
                    batch_size=batch_size,
                    epochs=100,
                    validation_data=(X_val.reshape(-1, X_val.shape[1], 1).astype('float32'), y_val.astype('int32')),
                    callbacks=[early_stopping, reduce_lr],
                    class_weight=cw,
                    verbose=2
                )
                
                val_acc = max(history.history.get('val_accuracy', [0]))
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = {
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'conv_layers': n_conv_layers
                    }
                    best_model = model
                    best_history = history
                
                print(f"Validation accuracy: {val_acc:.4f}")
    
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return best_model, best_params, best_history

def create_visualizations(results_df, history, model_name, output_dir='./output'):
    """Create comprehensive visualizations for CNN model analysis"""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Model Performance Comparison
    ax1 = plt.subplot(3, 3, 1)
    metrics = ['Train_Accuracy', 'Test_Accuracy', 'Val_Accuracy']
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
    
    # 2. Training History
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Training History', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Hyperparameter Performance
    ax3 = plt.subplot(3, 3, 3)
    param_cols = ['learning_rate', 'batch_size', 'conv_layers']
    param_values = results_df[param_cols].iloc[0]
    param_names = [col.replace('_', ' ').title() for col in param_cols]
    
    # Normalize values for comparison
    normalized_values = [
        param_values['learning_rate'] * 1000,  # Scale up learning rate
        param_values['batch_size'] / 100,      # Normalize batch size
        param_values['conv_layers'] / 5        # Normalize conv layers
    ]
    
    bars = ax3.bar(param_names, normalized_values, color='lightcoral')
    ax3.set_title('Hyperparameter Values (Normalized)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Normalized Value')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Loss History
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(history.history['loss'], label='Training Loss')
    ax4.plot(history.history['val_loss'], label='Validation Loss')
    ax4.set_title('Model Loss History', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Model Architecture Summary
    ax5 = plt.subplot(3, 3, 5)
    ax5.axis('off')
    architecture_text = [
        f"Conv Layers: {param_values['conv_layers']}",
        f"Dense Layers: 2",
        f"Dropout Rate: 0.5",
        f"Input Shape: {results_df['input_shape'].iloc[0]}",
        f"Output Classes: {results_df['num_classes'].iloc[0]}"
    ]
    ax5.text(0.1, 0.5, '\n'.join(architecture_text), 
             transform=ax5.transAxes, fontsize=12)
    ax5.set_title('Model Architecture', fontsize=14, fontweight='bold')
    
    # 6. Training Metrics
    ax6 = plt.subplot(3, 3, 6)
    metrics_text = [
        f"Total Epochs: {len(history.history['loss'])}",
        f"Best Val Accuracy: {max(history.history['val_accuracy']):.4f}",
        f"Final Val Loss: {history.history['val_loss'][-1]:.4f}",
        f"Training Time: {results_df['Training_Time'].iloc[0]:.2f}s"
    ]
    ax6.axis('off')
    ax6.text(0.1, 0.5, '\n'.join(metrics_text), 
             transform=ax6.transAxes, fontsize=12)
    ax6.set_title('Training Metrics', fontsize=14, fontweight='bold')
    
    # 7. Model Complexity
    ax7 = plt.subplot(3, 3, 7)
    model_params = results_df['total_params'].iloc[0]
    ax7.bar(['Model Parameters'], [model_params], color='orange')
    ax7.set_ylabel('Number of Parameters')
    ax7.set_title('Model Complexity', fontsize=14, fontweight='bold')
    ax7.text(0, model_params + model_params*0.05, f'{model_params:,}', 
             ha='center', fontweight='bold')
    
    # 8. Performance Summary Table
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    summary_data = [
        ['Metric', 'Value'],
        ['Train Accuracy', f"{results_df['Train_Accuracy'].iloc[0]:.3f}"],
        ['Test Accuracy', f"{results_df['Test_Accuracy'].iloc[0]:.3f}"],
        ['Val Accuracy', f"{results_df['Val_Accuracy'].iloc[0]:.3f}"],
        ['Training Time', f"{results_df['Training_Time'].iloc[0]:.2f}s"],
        ['Learning Rate', f"{results_df['learning_rate'].iloc[0]:.6f}"],
        ['Batch Size', f"{results_df['batch_size'].iloc[0]}"],
        ['Conv Layers', f"{results_df['conv_layers'].iloc[0]}"]
    ]
    
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
    
    # 9. Validation Metrics Distribution
    ax9 = plt.subplot(3, 3, 9)
    val_metrics = np.array(history.history['val_accuracy'])
    ax9.hist(val_metrics, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax9.axvline(np.mean(val_metrics), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(val_metrics):.3f}')
    ax9.set_xlabel('Validation Accuracy')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Validation Metrics Distribution', fontsize=14, fontweight='bold')
    ax9.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()

def debug_data(X, y):
    # basic label/feature sanity checks
    print("DEBUG: X shape", X.shape)
    print("DEBUG: y shape", y.shape)
    print("DEBUG: y unique (before):", np.unique(y))
    print("DEBUG: y min/max:", np.min(y), np.max(y))
    print("DEBUG: any NaN in X:", np.isnan(X).any())
    print("DEBUG: feature means (first 10):", np.nanmean(X, axis=0)[:10])
    print("DEBUG: feature stds (first 10):", np.nanstd(X, axis=0)[:10])
    const_feats = np.where(np.nanstd(X, axis=0) == 0)[0]
    print(f"DEBUG: number of constant features: {len(const_feats)}")
    # label distribution
    uniques, counts = np.unique(y, return_counts=True)
    print("DEBUG: label distribution:", dict(zip(uniques, counts)))
    return

def overfit_test(X, y, samples=32, epochs=200):
    # try to overfit small subset — should reach ~100% train acc if training pipeline correct
    idx = np.random.choice(len(X), min(samples, len(X)), replace=False)
    Xs = X[idx].reshape(-1, X.shape[1], 1).astype('float32')
    ys = y[idx].astype('int32')
    print(f"DEBUG: Running overfit test on {len(idx)} samples")
    model = create_cnn_model(input_shape=(X.shape[1], 1), num_classes=len(np.unique(y)), n_conv_layers=1)
    model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # reduce dropout for overfit test (if your model has fixed dropout, you can temporarily set rate=0.0)
    history = model.fit(Xs, ys, epochs=epochs, batch_size=min(8, len(idx)), verbose=2)
    final_acc = history.history['accuracy'][-1]
    print(f"DEBUG: Overfit final train accuracy: {final_acc:.4f}")
    return model, history

def main():
    """Main function for CNN-based ECG analysis"""
    
    print("="*80)
    print("CNN-BASED ECG ANALYSIS - HYPERPARAMETER TUNING & VISUALIZATION")
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
        # 1. Load dataset and extract features
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
            
            batch_start_time = time_module.time()
            
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
        
    # Adjust labels to be zero-based and integer
    y = (y - 1).astype(np.int32)
    
    # === FIX: Impute NaNs, drop constant features, ensure finite, then scale ===
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)  # replace NaNs with column means

    # remove constant / near-constant features (no signal)
    stds = np.nanstd(X, axis=0)
    keep_mask = stds > 1e-8
    if not np.all(keep_mask):
        removed = np.where(~keep_mask)[0]
        print(f"Removed {len(removed)} constant/near-constant features: {removed}")
        X = X[:, keep_mask]

    # Guard against any remaining non-finite values
    if not np.isfinite(X).all():
        print("Non-finite values detected after imputation — replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features (very important for NN convergence)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype('float32')
    # RUN DEBUG CHECKS
    debug_data(X, y)
    # Run overfit test — comment out later
    of_model, of_history = overfit_test(X, y, samples=32, epochs=100)
    print(f"\nFeature extraction completed!")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Number of age groups: {len(np.unique(y))}")
    print(f"  Age group labels: {np.unique(y)}")

    # === Merge very-rare ordinal classes into nearest neighbor classes ===
    def merge_rare_neighbors(y_arr, min_support=10, verbose=True):
        """
        Iteratively merge classes with support < min_support into the nearest neighbor class
        (by ordinal index). Prefer neighbor with larger count; tie -> right neighbor.
        Returns reindexed labels 0..K-1.
        """
        y_work = y_arr.copy().astype(int)
        # Guard: if only one class present, nothing to do
        if len(np.unique(y_work)) <= 1:
            return y_work

        # loop until all classes meet min_support or only one class remains
        while True:
            counts = Counter(y_work)
            # find smallest-count class (tie -> smaller class index)
            smallest_cls, smallest_cnt = min(counts.items(), key=lambda x: (x[1], x[0]))
            if smallest_cnt >= min_support:
                break

            classes_sorted = sorted(counts.keys())
            idx = classes_sorted.index(smallest_cls)
            left = classes_sorted[idx - 1] if idx - 1 >= 0 else None
            right = classes_sorted[idx + 1] if idx + 1 < len(classes_sorted) else None

            left_cnt = counts[left] if left is not None else -1
            right_cnt = counts[right] if right is not None else -1

            if left is None and right is None:
                # single remaining class
                break

            if right is None:
                target = left
            elif left is None:
                target = right
            else:
                # prefer neighbor with larger support (tie -> right)
                target = right if right_cnt >= left_cnt else left

            if verbose:
                print(f"Merging class {smallest_cls} (n={smallest_cnt}) -> neighbor {target} (n={counts[target]})")

            y_work[y_work == smallest_cls] = target
            # continue until condition met

        # reindex labels to contiguous 0..K-1
        unique = np.unique(y_work)
        mapping = {old: i for i, old in enumerate(unique)}
        y_reindexed = np.array([mapping[int(v)] for v in y_work], dtype=np.int32)
        return y_reindexed

    min_support = 12  # tune (e.g. 8,12,15)
    y = merge_rare_neighbors(y, min_support=min_support, verbose=True)
    print("New label distribution:", dict(Counter(y)))
    
    # 2. Split data
    print("\n2. Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Skip oversampling to avoid train-only memorization. Use class weights instead.
    try:
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = {int(cls): float(w) for cls, w in zip(np.unique(y_train), weights)}
        print("Using class weights:", class_weights)
    except Exception:
        class_weights = None
        print("Failed to compute class weights; proceeding without them.")
    
    # 3. Hyperparameter tuning
    print("\n3. Running hyperparameter tuning...")
    tuning_start_time = time_module.time()
    best_model, best_params, best_history = run_hyperparameter_tuning(X_train, y_train, X_val, y_val, class_weights=class_weights)
    training_time = time_module.time() - tuning_start_time
    
    # 4. Evaluate model
    print("\n4. Evaluating model...")
    
    # Reshape data for CNN
    X_train_reshaped = X_train.reshape(-1, X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(-1, X_test.shape[1], 1)
    X_val_reshaped = X_val.reshape(-1, X_val.shape[1], 1)
    
    # Predictions
    y_train_pred = np.argmax(best_model.predict(X_train_reshaped), axis=1)
    y_test_pred = np.argmax(best_model.predict(X_test_reshaped), axis=1)
    y_val_pred = np.argmax(best_model.predict(X_val_reshaped), axis=1)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"\nModel Performance:")
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Training Time: {training_time:.2f}s")
    
    # 5. Save results
    results_df = pd.DataFrame({
        'Model': ['CNN_Tuned'],
        'Train_Accuracy': [train_accuracy],
        'Test_Accuracy': [test_accuracy],
        'Val_Accuracy': [val_accuracy],
        'Training_Time': [training_time],
        'learning_rate': [best_params['learning_rate']],
        'batch_size': [best_params['batch_size']],
        'conv_layers': [best_params['conv_layers']],
        'input_shape': [str(X_train.shape[1:])],
        'num_classes': [len(np.unique(y))],
        'total_params': [best_model.count_params()]
    })
    
    # Save results to CSV
    results_path = os.path.join(output_dir, 'cnn_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # 6. Save model
    model_path = os.path.join(output_dir, 'cnn_model.keras')
    best_model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Optional: Save in h5 format for older versions compatibility
    h5_model_path = os.path.join(output_dir, 'cnn_model.h5')
    best_model.save(h5_model_path)
    print(f"Model also saved in H5 format to: {h5_model_path}")
    
    # 7. Classification report
    print("\n7. Detailed Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # 8. Create visualizations
    print("\n8. Creating visualizations...")
    create_visualizations(results_df, best_history, 'CNN_Tuned', output_dir)
    
    print("\n" + "="*80)
    print("CNN ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()