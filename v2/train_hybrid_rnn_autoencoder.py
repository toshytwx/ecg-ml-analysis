#!/usr/bin/env python3
"""
Hybrid RNN + Autoencoder Training Script
========================================

This script implements a hybrid architecture that combines:
1. Pre-trained Autoencoder for feature extraction
2. RNN for temporal sequence modeling
3. Classification head for age group prediction

The approach:
- Use autoencoder to extract 64-dimensional features from ECG segments
- Feed these features to an RNN for temporal modeling
- Classify age groups using the RNN's final hidden state
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configuration
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LATENT_DIM = 64  # Autoencoder latent dimension
SEQUENCE_LENGTH = 10  # Number of segments per ECG

# Training configuration
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
PATIENCE = 8

# Data configuration
DATA_DIR = "input/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0"
SIGNAL_LENGTH = 2500
SEGMENT_LENGTH = 250  # Each segment is 250 samples
NUM_SEGMENTS = SIGNAL_LENGTH // SEGMENT_LENGTH  # 10 segments per ECG

# Output configuration
OUTPUT_DIR = "ecg_hybrid_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_ecg_data():
    """Load ECG data and create segments for hybrid model"""
    print("🔄 Loading ECG data...")
    
    # Load subject info
    subject_info_path = os.path.join(DATA_DIR, "subject-info.csv")
    subject_info = pd.read_csv(subject_info_path)
    print(f"Loaded subject info: {len(subject_info)} subjects")
    
    # Get available age groups
    age_groups = subject_info['Age_group'].dropna().unique()
    print(f"Age groups available: {sorted(age_groups)}")
    
    # Load ECG files using .hea files (like wfdb_parser)
    import glob
    hea_files = glob.glob(os.path.join(DATA_DIR, "*.hea"))
    hea_files.sort()
    
    print(f"Found {len(hea_files)} ECG files")
    
    # Load signals and create segments
    signals = []
    ages = []
    
    print("Loading signals and creating segments...")
    for hea_file in tqdm(hea_files):  # Load all files
        try:
            # Get record name from .hea file
            record_name = os.path.basename(hea_file).replace('.hea', '')
            
            # Load signal from corresponding .dat file
            dat_file = os.path.join(DATA_DIR, record_name + '.dat')
            if not os.path.exists(dat_file):
                continue
                
            signal = np.fromfile(dat_file, dtype=np.int16)
            
            # Normalize signal
            signal = signal.astype(np.float32)
            signal = (signal - signal.mean()) / (signal.std() + 1e-8)
            
            # Create segments
            segments = []
            for i in range(0, len(signal) - SEGMENT_LENGTH + 1, SEGMENT_LENGTH):
                segment = signal[i:i + SEGMENT_LENGTH]
                if len(segment) == SEGMENT_LENGTH:
                    segments.append(segment)
            
            # Take first 10 segments
            segments = segments[:NUM_SEGMENTS]
            if len(segments) == NUM_SEGMENTS:
                # Get age group using the same logic as wfdb_parser
                record_id = record_name.lstrip("0")
                if record_id == "":
                    record_id = "0"
                
                # Check if record_id exists in the DataFrame
                if record_id in subject_info['ID'].astype(str).values:
                    age_group = subject_info[subject_info['ID'].astype(str) == record_id]['Age_group'].iloc[0]
                else:
                    age_group = 3  # Default age group
                
                if not pd.isna(age_group):
                    signals.append(np.array(segments))
                    ages.append(int(age_group))
                    
        except Exception as e:
            continue
    
    signals = np.array(signals)
    ages = np.array(ages)
    
    print(f"Loaded {len(signals)} ECG sequences")
    print(f"Signal shape: {signals.shape}")
    print(f"Age groups: {np.unique(ages)}")
    print(f"Age distribution: {np.bincount(ages)}")
    
    return signals, ages

# =============================================================================
# HYBRID MODEL ARCHITECTURE
# =============================================================================

class HybridRNNAutoencoder(nn.Module):
    """Hybrid RNN + Autoencoder model"""
    
    def __init__(self, input_dim, hidden_size, num_layers, latent_dim, n_classes, dropout=0.3):
        super(HybridRNNAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        
        # Autoencoder encoder (pre-trained)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim)
        )
        
        # RNN for temporal modeling
        self.rnn = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, segment_length)
        batch_size, seq_len, segment_len = x.shape
        
        # Reshape for autoencoder: (batch_size * sequence_length, 1, segment_length)
        x_reshaped = x.view(-1, 1, segment_len)
        
        # Extract features using autoencoder encoder
        features = self.encoder(x_reshaped)  # (batch_size * seq_len, latent_dim)
        
        # Reshape back to sequence: (batch_size, seq_len, latent_dim)
        features = features.view(batch_size, seq_len, self.latent_dim)
        
        # RNN processing
        rnn_out, (hidden, cell) = self.rnn(features)
        
        # Use the last output for classification
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # Classification
        output = self.classifier(last_output)
        
        return output

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Memory management
        if batch_idx % 100 == 0:
            torch.mps.empty_cache()
    
    return total_loss / len(train_loader), correct / total

def validate_epoch(model, val_loader, criterion, device):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(val_loader), correct / total

def train_hybrid_model(model, train_loader, val_loader, epochs, lr, patience, device):
    """Train the hybrid model"""
    print(f"\n🎯 Training Hybrid RNN + Autoencoder Model")
    print("=" * 50)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_hybrid_model.pth'))
            print(f"✅ New best model saved! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
        
        # Memory management
        torch.mps.empty_cache()
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }

# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_visualizations(results, model, test_loader, device):
    """Generate training visualizations"""
    print("\n📊 Generating visualizations...")
    
    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(results['train_losses'], label='Train Loss', color='blue')
    axes[0, 0].plot(results['val_losses'], label='Val Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(results['train_accs'], label='Train Acc', color='blue')
    axes[0, 1].plot(results['val_accs'], label='Val Acc', color='red')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Test predictions
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Class distribution
    unique, counts = np.unique(all_targets, return_counts=True)
    axes[1, 1].bar(unique, counts)
    axes[1, 1].set_title('Class Distribution')
    axes[1, 1].set_xlabel('Age Group')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_training_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {OUTPUT_DIR}/hybrid_training_analysis.png")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main training function"""
    print("🧠 Hybrid RNN + Autoencoder Training")
    print("=" * 40)
    
    # Load data
    signals, ages = load_ecg_data()
    
    # Create age groups (use ages directly as they're already age groups)
    age_groups = ages - 1  # Convert to 0-based indexing
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  • Total samples: {len(signals)}")
    print(f"  • Age groups: {len(np.unique(age_groups))}")
    print(f"  • Class distribution: {np.bincount(age_groups)}")
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        signals, age_groups, test_size=0.2, random_state=42, stratify=age_groups
    )
    
    print(f"\n📊 Train/Validation Split:")
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Validation samples: {len(X_val)}")
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    n_classes = len(np.unique(age_groups))
    model = HybridRNNAutoencoder(
        input_dim=SEGMENT_LENGTH,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        latent_dim=LATENT_DIM,
        n_classes=n_classes,
        dropout=DROPOUT
    ).to(device)
    
    print(f"\n🏗️ Model Architecture:")
    print(f"  • Input dimension: {SEGMENT_LENGTH}")
    print(f"  • Sequence length: {NUM_SEGMENTS}")
    print(f"  • Latent dimension: {LATENT_DIM}")
    print(f"  • Hidden size: {HIDDEN_SIZE}")
    print(f"  • Number of layers: {NUM_LAYERS}")
    print(f"  • Number of classes: {n_classes}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  • Total parameters: {total_params:,}")
    print(f"  • Trainable parameters: {trainable_params:,}")
    
    # Train model
    start_time = time.time()
    results = train_hybrid_model(model, train_loader, val_loader, EPOCHS, LR, PATIENCE, device)
    training_time = time.time() - start_time
    
    print(f"\n⏱️ Training completed in {training_time:.2f} seconds")
    print(f"🏆 Best validation accuracy: {results['best_val_acc']:.4f}")
    
    # Generate visualizations
    generate_visualizations(results, model, val_loader, device)
    
    # Save results
    results_summary = {
        'model_type': 'Hybrid RNN + Autoencoder',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'best_val_accuracy': results['best_val_acc'],
        'training_time': training_time,
        'epochs_trained': len(results['train_losses']),
        'final_train_loss': results['train_losses'][-1],
        'final_val_loss': results['val_losses'][-1],
        'final_train_acc': results['train_accs'][-1],
        'final_val_acc': results['val_accs'][-1]
    }
    
    with open(os.path.join(OUTPUT_DIR, 'hybrid_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Results saved to: {OUTPUT_DIR}/")
    print(f"📊 Generated files:")
    print(f"  • hybrid_training_analysis.png")
    print(f"  • hybrid_results.json")
    print(f"  • best_hybrid_model.pth")

if __name__ == "__main__":
    main()
