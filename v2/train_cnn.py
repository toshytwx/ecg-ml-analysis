#!/usr/bin/env python3
"""
train_cnn_ecg_advanced.py

Advanced 1D CNN for ECG age group classification with:
- Class imbalance handling (focal loss, class weighting, data augmentation)
- Advanced architecture (ResNet blocks, attention mechanisms)
- Comprehensive regularization and training strategies
- Direct WFDB parsing from .dat/.hea files
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import random
from collections import Counter
import warnings
from wfdb_parser import create_wfdb_dataset
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
DATA_PATH = "/Users/dmytro/Diploma/ecg_ml_analysis/v2/input/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0"
SUBJECT_INFO_CSV = "/Users/dmytro/Diploma/ecg_ml_analysis/v2/input/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0/subject-info.csv"
OUTPUT_DIR = "./ecg_cnn_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset parameters
WINDOW_SIZE_SEC = 10
WINDOW_STEP_SEC = 5
PRELOAD_DATA = True  # Optimized for 16GB RAM

# Training parameters
RANDOM_STATE = 42
TEST_SPLIT = 0.2
BATCH_SIZE = 128  # Optimized for M4 Pro GPU
EPOCHS = 100
LR = 1e-3

# Device detection with M4 Pro GPU support
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # M4 Pro GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    else:
        return torch.device("cpu")  # CPU fallback

DEVICE = get_device()
print(f"Using device: {DEVICE}")

# M4 Pro GPU optimizations
if DEVICE.type == "mps":
    torch.backends.mps.allow_tf32 = True
    torch.backends.mps.allow_fp16 = True
    print("M4 Pro GPU optimizations enabled")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------- ADVANCED MODEL ARCHITECTURE ----------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels//4, 1),
            nn.ReLU(),
            nn.Conv1d(channels//4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        att = self.attention(x)
        return x * att

class AdvancedCNN1D(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Residual blocks with attention
        self.res_block1 = ResidualBlock(64, 64)
        self.attention1 = AttentionBlock(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.res_block2 = ResidualBlock(64, 128, stride=2)
        self.attention2 = AttentionBlock(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.res_block3 = ResidualBlock(128, 256, stride=2)
        self.attention3 = AttentionBlock(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, n_classes)
        
    def forward(self, x):
        # x: (B, C, L)
        x = self.initial_conv(x)
        
        x = self.res_block1(x)
        x = self.attention1(x)
        x = self.pool1(x)
        
        x = self.res_block2(x)
        x = self.attention2(x)
        x = self.pool2(x)
        
        x = self.res_block3(x)
        x = self.attention3(x)
        x = self.pool3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def main():
    """Main training function."""
    # ---------------- LOAD DATASET ----------------
    print("Loading dataset from WFDB files...")
    dataset = create_wfdb_dataset(
        data_path=DATA_PATH,
        subject_info_csv=SUBJECT_INFO_CSV,
        window_size_sec=WINDOW_SIZE_SEC,
        window_step_sec=WINDOW_STEP_SEC,
        augment=False,  # We'll handle augmentation in the training loop
        preload=PRELOAD_DATA
    )

    # Train/test split
    test_size = int(len(dataset) * TEST_SPLIT)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create a wrapper for the training dataset that enables augmentation
    class AugmentedDataset:
        def __init__(self, base_dataset, indices):
            self.base_dataset = base_dataset
            self.indices = indices
            # Enable augmentation for this dataset
            self.base_dataset.augment = True
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.base_dataset[self.indices[idx]]

    # Create augmented training dataset
    train_dataset_aug = AugmentedDataset(dataset, train_dataset.indices)

    n_classes = len(dataset.classes_)
    n_channels = dataset.max_channels
    print(f"Number of classes: {n_classes}, Channels: {n_channels}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Augmented train samples: {len(train_dataset_aug)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Debug: Check if indices are valid
    print(f"Dataset total samples: {len(dataset)}")
    print(f"Train indices range: {min(train_dataset.indices)} to {max(train_dataset.indices)}")
    print(f"Test indices range: {min(test_dataset.indices)} to {max(test_dataset.indices)}")

    # ---------------- COMPUTE CLASS WEIGHTS ----------------
    # Get class distribution for weighting
    all_labels = []
    for sample in dataset.samples:
        all_labels.append(dataset.le.transform([sample['age_group']])[0])

    class_counts = Counter(all_labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)

    print(f"Class distribution: {dict(class_counts)}")
    print(f"Class weights: {class_weights}")

    # ---------------- INITIALIZE MODEL AND TRAINING ----------------
    model = AdvancedCNN1D(n_channels, n_classes).to(DEVICE)

    # Use weighted sampler for training (only for training indices)
    train_labels = [all_labels[i] for i in train_dataset.indices]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create data loaders (use num_workers=0 to avoid multiprocessing issues on macOS)
    train_loader = DataLoader(train_dataset_aug, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Advanced optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # Use focal loss for better handling of class imbalance
    criterion = FocalLoss(alpha=1, gamma=2)

    # ---------------- ADVANCED TRAINING WITH EARLY STOPPING ----------------
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0
    patience = 20
    patience_counter = 0

    for epoch in range(1, EPOCHS+1):
        # Training
        print(f"Training epoch {epoch}/{EPOCHS}")
        if DEVICE.type == "mps":
            print(f"M4 Pro GPU memory allocated: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        
        train_losses.append(running_loss/total)
        train_accs.append(correct/total)

        # Validation
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                running_loss += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        
        val_losses.append(running_loss/total)
        val_accs.append(correct/total)
        
        # Learning rate scheduling
        scheduler.step(val_accs[-1])
        
        # Early stopping
        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch}/{EPOCHS} | Train Acc: {train_accs[-1]:.3f} | Val Acc: {val_accs[-1]:.3f} | Best: {best_val_acc:.3f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))

    # ---------------- COMPREHENSIVE EVALUATION ----------------
    model.eval()
    y_true, y_pred = [], []
    probs_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            probs = nn.functional.softmax(out, dim=1)
            max_probs, _ = probs.max(1)
            probs_list.append(max_probs.cpu().numpy())

            preds = out.argmax(dim=1)
            y_true.append(yb.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    probs_all = np.concatenate(probs_list)

    # Comprehensive metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print(f"\n=== COMPREHENSIVE EVALUATION ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    print("\nDetailed Classification Report:\n")
    print(classification_report(y_true, y_pred, digits=4, target_names=[f'Age_{c}' for c in dataset.le.classes_]))
    cm = confusion_matrix(y_true, y_pred)

    # ---------------- COMPREHENSIVE PLOTS ----------------
    sns.set(style="whitegrid")

    # 1) Training curves with learning rate
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0,0].plot(train_losses, label="Train Loss")
    axes[0,0].plot(val_losses, label="Val Loss")
    axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("Loss")
    axes[0,0].set_title("Loss Curves"); axes[0,0].legend(); axes[0,0].grid(True)

    axes[0,1].plot(train_accs, label="Train Acc")
    axes[0,1].plot(val_accs, label="Val Acc")
    axes[0,1].set_xlabel("Epoch"); axes[0,1].set_ylabel("Accuracy")
    axes[0,1].set_title("Accuracy Curves"); axes[0,1].legend(); axes[0,1].grid(True)

    # Class distribution
    class_dist = Counter(y_true)
    axes[1,0].bar(range(len(class_dist)), list(class_dist.values()))
    axes[1,0].set_xlabel("Age Group"); axes[1,0].set_ylabel("Count")
    axes[1,0].set_title("Test Set Class Distribution")
    axes[1,0].set_xticks(range(len(class_dist)))
    axes[1,0].set_xticklabels([f'Age_{c}' for c in sorted(class_dist.keys())], rotation=45)

    # Prediction confidence
    axes[1,1].hist(probs_all, bins=20, color="green", alpha=0.7)
    axes[1,1].set_xlabel("Max Predicted Probability")
    axes[1,1].set_ylabel("Count")
    axes[1,1].set_title("Prediction Confidence Distribution")
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_analysis.png"))

    # 2) Enhanced confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f'Age_{c}' for c in dataset.le.classes_], 
                yticklabels=[f'Age_{c}' for c in dataset.le.classes_])
    plt.xlabel("Predicted Age Group"); plt.ylabel("True Age Group")
    plt.title(f"Confusion Matrix\nAccuracy: {acc:.3f} | F1-Macro: {f1_macro:.3f} | F1-Weighted: {f1_weighted:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # 3) Per-class performance analysis
    per_class_acc = []
    per_class_f1 = []
    for i in range(n_classes):
        idx = (y_true == i)
        if idx.any():
            acc_i = (y_pred[idx] == y_true[idx]).mean()
            # For multiclass, we need to use a different approach for per-class F1
            # Create binary labels for this class vs all others
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            f1_i = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        else:
            acc_i = 0.0
            f1_i = 0.0
        per_class_acc.append(acc_i)
        per_class_f1.append(f1_i)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    x_pos = range(len(dataset.le.classes_))
    ax1.bar(x_pos, per_class_acc, color="skyblue", alpha=0.7)
    ax1.set_xlabel("Age Group"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Per-class Accuracy")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Age_{c}' for c in dataset.le.classes_], rotation=45)
    ax1.grid(axis='y')

    ax2.bar(x_pos, per_class_f1, color="lightcoral", alpha=0.7)
    ax2.set_xlabel("Age Group"); ax2.set_ylabel("F1-Score")
    ax2.set_title("Per-class F1-Score")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Age_{c}' for c in dataset.le.classes_], rotation=45)
    ax2.grid(axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "per_class_performance.png"))

    # 4) Class imbalance analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # True vs Predicted distribution
    ax1.hist(y_true, bins=len(dataset.le.classes_), alpha=0.7, label="True", color="blue")
    ax1.hist(y_pred, bins=len(dataset.le.classes_), alpha=0.7, label="Predicted", color="red")
    ax1.set_xlabel("Age Group"); ax1.set_ylabel("Count")
    ax1.set_title("True vs Predicted Distribution")
    ax1.legend(); ax1.grid(True)

    # Class weights visualization
    class_weights_np = class_weights.cpu().numpy()
    ax2.bar(range(len(class_weights_np)), class_weights_np, color="orange", alpha=0.7)
    ax2.set_xlabel("Age Group"); ax2.set_ylabel("Class Weight")
    ax2.set_title("Computed Class Weights")
    ax2.set_xticks(range(len(dataset.le.classes_)))
    ax2.set_xticklabels([f'Age_{c}' for c in dataset.le.classes_], rotation=45)
    ax2.grid(axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_imbalance_analysis.png"))

    # 5) Sample ECG signals with predictions
    plt.figure(figsize=(15, 10))
    sample_indices = np.random.choice(len(test_dataset), 6, replace=False)
    for i, idx in enumerate(sample_indices):
        xb, yb = test_dataset[idx]
        xb_numpy = xb.cpu().numpy()  # Keep numpy version for plotting
        y_true_sample = dataset.le.inverse_transform([yb.item()])[0]
        with torch.no_grad():
            # Use the original tensor for model inference
            out = model(xb.unsqueeze(0).to(DEVICE))
            pred_label = dataset.le.inverse_transform([out.argmax(1).item()])[0]
            confidence = torch.softmax(out, dim=1).max().item()
        
        plt.subplot(3, 2, i+1)
        plt.plot(xb_numpy.T, alpha=0.7)  # plot all channels
        plt.title(f"True: Age_{y_true_sample} | Pred: Age_{pred_label} | Conf: {confidence:.3f}")
        plt.xlabel("Time"); plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_ecg_predictions.png"))

    # 6) Model architecture visualization (simplified)
    plt.figure(figsize=(12, 8))
    plt.text(0.1, 0.9, "Advanced CNN Architecture:", fontsize=16, fontweight='bold')
    plt.text(0.1, 0.8, "• Residual Blocks with Skip Connections", fontsize=12)
    plt.text(0.1, 0.75, "• Attention Mechanisms for Feature Selection", fontsize=12)
    plt.text(0.1, 0.7, "• Batch Normalization and Dropout", fontsize=12)
    plt.text(0.1, 0.65, "• Focal Loss for Class Imbalance", fontsize=12)
    plt.text(0.1, 0.6, "• Weighted Random Sampling", fontsize=12)
    plt.text(0.1, 0.55, "• Learning Rate Scheduling", fontsize=12)
    plt.text(0.1, 0.5, "• Early Stopping", fontsize=12)
    plt.text(0.1, 0.4, f"Final Performance:", fontsize=14, fontweight='bold')
    plt.text(0.1, 0.35, f"• Accuracy: {acc:.4f}", fontsize=12)
    plt.text(0.1, 0.3, f"• F1-Macro: {f1_macro:.4f}", fontsize=12)
    plt.text(0.1, 0.25, f"• F1-Weighted: {f1_weighted:.4f}", fontsize=12)
    plt.text(0.1, 0.2, f"• Classes: {n_classes}", fontsize=12)
    plt.text(0.1, 0.15, f"• Channels: {n_channels}", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_summary.png"))

    print("All comprehensive plots saved to:", OUTPUT_DIR)

if __name__ == '__main__':
    main()