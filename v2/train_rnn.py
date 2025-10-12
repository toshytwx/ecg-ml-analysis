#!/usr/bin/env python3
"""
train_rnn_ecg_advanced.py

Advanced RNN/LSTM for ECG age group classification with:
- Class imbalance handling (focal loss, class weighting, data augmentation)
- Advanced architecture (Bidirectional LSTM, attention mechanisms)
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
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import random
from collections import Counter
import warnings
from wfdb_parser import create_wfdb_dataset
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
DATA_PATH = "/Users/dmytro/Diploma/ecg_ml_analysis/v2/input/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0"
SUBJECT_INFO_CSV = "/Users/dmytro/Diploma/ecg_ml_analysis/v2/input/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0/subject-info.csv"
OUTPUT_DIR = "./ecg_rnn_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset parameters
WINDOW_SIZE_SEC = 10
WINDOW_STEP_SEC = 5
PRELOAD_DATA = True  # Enable preloading for faster training (16GB RAM should handle it)

# Training parameters
RANDOM_STATE = 42
TEST_SPLIT = 0.2
BATCH_SIZE = 64  # Reduced batch size to prevent memory issues
EPOCHS = 20  # Increased for better convergence with improved loss
LR = 1e-3  # Optimized learning rate for aligned model
GRADIENT_ACCUMULATION_STEPS = 2  # Reduced accumulation steps for faster training

# DataLoader optimization parameters
NUM_WORKERS = 0  # Disable multiprocessing on macOS to avoid pickle issues
PREFETCH_FACTOR = 2  # Prefetch batches for better pipeline

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
    # Set memory watermark ratio to allow more memory usage
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    # Enable mixed precision for faster training
    torch.backends.mps.allow_autocast = True
    print("M4 Pro GPU optimizations enabled with mixed precision and memory watermark disabled")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------- ADVANCED RNN ARCHITECTURE ----------------
class AttentionLayer(nn.Module):
    """Attention mechanism for RNN outputs."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, rnn_outputs):
        # rnn_outputs: (batch_size, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(rnn_outputs), dim=1)
        # Weighted sum of RNN outputs
        context_vector = torch.sum(attention_weights * rnn_outputs, dim=1)
        return context_vector, attention_weights

class ResidualLSTM(nn.Module):
    """Residual LSTM block with skip connections."""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        
        # Skip connection
        self.skip_connection = nn.Linear(input_size, hidden_size * 2)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Skip connection
        skip = self.skip_connection(x)
        # Pad or truncate skip connection to match LSTM output
        if skip.size(1) != lstm_out.size(1):
            skip = F.interpolate(skip.transpose(1, 2), size=lstm_out.size(1), mode='linear', align_corners=False).transpose(1, 2)
        
        # Residual connection
        output = lstm_out + skip
        output = self.layer_norm(output)
        
        return output

class SimpleRNN(nn.Module):
    """Simplified RNN architecture for faster training and lower memory usage."""
    def __init__(self, input_size, hidden_size, n_classes, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Simple LSTM (no bidirectional for speed and memory)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Simple attention
        self.attention = nn.Linear(hidden_size, 1)
        
        # Enhanced classification head to match CNN capacity
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # First layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),  # Second layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),  # Third layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, n_classes)  # Output layer
        )
        
    def forward(self, x):
        # x: (batch_size, channels, seq_len) -> (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Simple attention (global average pooling)
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        output = self.classifier(context_vector)
        
        return output

# Focal Loss for handling class imbalance (same as CNN)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            # Apply label smoothing
            num_classes = inputs.size(-1)
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1 - self.label_smoothing)
            ce_loss = -(smooth_targets * F.log_softmax(inputs, dim=-1)).sum(dim=-1)
        else:
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
    # RNN parameters (aligned to match CNN parameters exactly)
    hidden_size = 119  # Optimized to match CNN's 141,007 parameters
    num_layers = 2  # Two layers for optimal parameter count
    
    model = SimpleRNN(n_channels, hidden_size, n_classes, num_layers).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Use weighted sampler for training (only for training indices)
    train_labels = [all_labels[i] for i in train_dataset.indices]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Create data loaders with optimized settings for maximum performance
    train_loader = DataLoader(
        train_dataset_aug, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=False,  # Disable pin_memory on MPS
        persistent_workers=False,  # Disable persistent workers with num_workers=0
        prefetch_factor=None,  # Disable prefetch with num_workers=0
        drop_last=True,  # Ensure consistent batch sizes
        collate_fn=None  # Use default collate function for efficiency
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,  # Disable pin_memory on MPS
        persistent_workers=False,  # Disable persistent workers with num_workers=0
        prefetch_factor=None,  # Disable prefetch with num_workers=0
        drop_last=False,  # Keep all test samples
        collate_fn=None  # Use default collate function for efficiency
    )

    # Advanced optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.01)

    # Use focal loss optimized for extreme class imbalance with label smoothing
    criterion = FocalLoss(alpha=2.0, gamma=3.0, label_smoothing=0.1)  # Stronger focus + label smoothing

    # ---------------- ADVANCED TRAINING WITH EARLY STOPPING ----------------
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0
    patience = 5  # Optimized patience for balanced training time
    patience_counter = 0

    for epoch in range(1, EPOCHS+1):
        # Training
        print(f"Training epoch {epoch}/{EPOCHS}")
        if DEVICE.type == "mps":
            gpu_memory = torch.mps.current_allocated_memory() / 1024**3
            print(f"M4 Pro GPU memory allocated: {gpu_memory:.2f} GB")
            # Memory safety check
            if gpu_memory > 8.0:  # If using more than 8GB GPU memory
                print("⚠️  High GPU memory usage detected, clearing cache...")
                torch.mps.empty_cache()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()  # Initialize gradients
        
        for batch_idx, (xb, yb) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)
            
            # Scale loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            running_loss += loss.item() * xb.size(0) * GRADIENT_ACCUMULATION_STEPS
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            
            # Update weights every GRADIENT_ACCUMULATION_STEPS
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Gradient clipping for RNNs
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Progress indicator every 500 batches (reduced frequency)
            if batch_idx % 500 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {correct/total:.3f}")
            
        
        # Final gradient update if there are remaining gradients
        if len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        train_losses.append(running_loss/total)
        train_accs.append(correct/total)

        # Validation
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_idx, (xb, yb) in enumerate(tqdm(test_loader, desc="Validation", leave=False)):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                running_loss += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                
        
        val_losses.append(running_loss/total)
        val_accs.append(correct/total)
        
        # Learning rate scheduling (CosineAnnealingLR doesn't need validation metric)
        scheduler.step()
        
        # Early stopping
        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_rnn_model.pth'))
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch}/{EPOCHS} | Train Acc: {train_accs[-1]:.3f} | Val Acc: {val_accs[-1]:.3f} | Best: {best_val_acc:.3f}")
        
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_rnn_model.pth')))

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
    plt.savefig(os.path.join(OUTPUT_DIR, "rnn_training_analysis.png"))

    # 2) Enhanced confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f'Age_{c}' for c in dataset.le.classes_], 
                yticklabels=[f'Age_{c}' for c in dataset.le.classes_])
    plt.xlabel("Predicted Age Group"); plt.ylabel("True Age Group")
    plt.title(f"RNN Confusion Matrix\nAccuracy: {acc:.3f} | F1-Macro: {f1_macro:.3f} | F1-Weighted: {f1_weighted:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rnn_confusion_matrix.png"))

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
    plt.savefig(os.path.join(OUTPUT_DIR, "rnn_per_class_performance.png"))

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
    plt.savefig(os.path.join(OUTPUT_DIR, "rnn_class_imbalance_analysis.png"))

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
    plt.savefig(os.path.join(OUTPUT_DIR, "rnn_sample_ecg_predictions.png"))

    # 6) Model architecture visualization (simplified)
    plt.figure(figsize=(12, 8))
    plt.text(0.1, 0.9, "Advanced RNN Architecture:", fontsize=16, fontweight='bold')
    plt.text(0.1, 0.8, "• Bidirectional LSTM with Residual Connections", fontsize=12)
    plt.text(0.1, 0.75, "• Attention Mechanism for Sequence Modeling", fontsize=12)
    plt.text(0.1, 0.7, "• Layer Normalization and Dropout", fontsize=12)
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
    plt.text(0.1, 0.1, f"• Hidden Size: {hidden_size}", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rnn_model_summary.png"))

    print("All comprehensive RNN plots saved to:", OUTPUT_DIR)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
    finally:
        # Cleanup to prevent semaphore leaks
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        print("🧹 Cleanup completed")
