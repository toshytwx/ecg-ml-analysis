#!/usr/bin/env python3
"""
ECG Autoencoder Training Script
==============================

This script implements a 1D Convolutional Autoencoder for ECG data:
- Encoder: Compresses ECG signals into latent representations
- Decoder: Reconstructs signals from latent space
- Applications: Denoising, feature learning, data augmentation

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import wfdb

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data configuration
DATA_DIR = "input/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0"
OUTPUT_DIR = "ecg_autoencoder_outputs"
SIGNAL_LENGTH = 2500  # 10 seconds at 250 Hz
SIGNAL_DOWNSAMPLE = 1  # Keep original sampling rate
PRELOAD_DATA = True
BATCH_SIZE = 64
NUM_WORKERS = 0  # Disable multiprocessing for macOS compatibility

# Model configuration
LATENT_DIM = 64  # Latent space dimension
ENCODER_CHANNELS = [1, 32, 64, 128]  # Encoder channel progression
DECODER_CHANNELS = [128, 64, 32, 1]  # Decoder channel progression
KERNEL_SIZE = 15
STRIDE = 1
PADDING = 7

# Training configuration
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 10
MIN_DELTA = 1e-4

# Loss configuration
MSE_WEIGHT = 1.0
L1_WEIGHT = 0.1

# Device configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

class ECGAutoencoderDataset(Dataset):
    """Dataset for ECG autoencoder training"""
    
    def __init__(self, signals, labels=None, transform=None):
        self.signals = signals
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        
        if self.transform:
            signal = self.transform(signal)
            
        if self.labels is not None:
            return signal, self.labels[idx]
        else:
            return signal, signal  # For autoencoder, input = target

def load_ecg_data():
    """Load and preprocess ECG data for autoencoder training"""
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
                    # Handle NaN values
                    if pd.isna(age_group):
                        ages.append(3)  # Default age group for NaN
                    else:
                        ages.append(int(age_group))
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
    print(f"Age distribution: {np.bincount(ages)}")
    
    return signals, ages

def create_age_groups(ages):
    """Create age groups for analysis"""
    age_groups = []
    for age in ages:
        if age < 30:
            age_groups.append(0)
        elif age < 40:
            age_groups.append(1)
        elif age < 50:
            age_groups.append(2)
        elif age < 60:
            age_groups.append(3)
        elif age < 70:
            age_groups.append(4)
        elif age < 80:
            age_groups.append(5)
        else:
            age_groups.append(6)
    return np.array(age_groups)

# =============================================================================
# AUTOENCODER MODEL
# =============================================================================

class ECGAutoencoder(nn.Module):
    """1D Convolutional Autoencoder for ECG signals"""
    
    def __init__(self, input_length, latent_dim, encoder_channels, decoder_channels, 
                 kernel_size, stride, padding):
        super(ECGAutoencoder, self).__init__()
        
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.ModuleList()
        current_length = input_length
        
        for i in range(len(encoder_channels) - 1):
            in_channels = encoder_channels[i]
            out_channels = encoder_channels[i + 1]
            
            self.encoder.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ))
            current_length = (current_length + 2 * padding - kernel_size) // stride + 1
        
        # Latent space projection
        self.latent_proj = nn.Linear(encoder_channels[-1] * current_length, latent_dim)
        
        # Decoder
        self.decoder_proj = nn.Linear(latent_dim, encoder_channels[-1] * current_length)
        
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_channels) - 1):
            in_channels = decoder_channels[i]
            out_channels = decoder_channels[i + 1]
            
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True) if i < len(decoder_channels) - 2 else nn.Identity()
            ))
        
        # Store dimensions for reconstruction
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.current_length = current_length
        
    def encode(self, x):
        """Encode input to latent space"""
        for layer in self.encoder:
            x = layer(x)
        
        # Flatten and project to latent space
        x = x.view(x.size(0), -1)
        z = self.latent_proj(x)
        return z
    
    def decode(self, z):
        """Decode latent representation to output"""
        # Project back to encoder output shape
        x = self.decoder_proj(z)
        x = x.view(x.size(0), self.encoder_channels[-1], self.current_length)
        
        # Decode through transposed convolutions
        for layer in self.decoder:
            x = layer(x)
        
        return x
    
    def forward(self, x):
        """Forward pass"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

def ssim_loss(x, y, window_size=11, sigma=1.5):
    """Structural Similarity Index (SSIM) loss - simplified version"""
    # Simplified SSIM loss to avoid device issues
    # Use MSE as fallback for now
    return F.mse_loss(x, y)

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    mse_loss = 0
    l1_loss = 0
    ssim_loss_val = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        recon_data, latent = model(data)
        
        # Calculate losses
        mse = F.mse_loss(recon_data, data)
        l1 = F.l1_loss(recon_data, data)
        
        loss = MSE_WEIGHT * mse + L1_WEIGHT * l1
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        mse_loss += mse.item()
        l1_loss += l1.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'MSE': f'{mse.item():.4f}',
            'L1': f'{l1.item():.4f}'
        })
        
        # Memory management
        if batch_idx % 100 == 0:
            if device.type == 'mps':
                torch.mps.empty_cache()
    
    return total_loss / len(train_loader), mse_loss / len(train_loader), l1_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    mse_loss = 0
    l1_loss = 0
    
    with torch.no_grad():
        for data, _ in tqdm(val_loader, desc="Validation", leave=False):
            data = data.to(device)
            
            recon_data, latent = model(data)
            
            mse = F.mse_loss(recon_data, data)
            l1 = F.l1_loss(recon_data, data)
            
            loss = MSE_WEIGHT * mse + L1_WEIGHT * l1
            
            total_loss += loss.item()
            mse_loss += mse.item()
            l1_loss += l1.item()
    
    return total_loss / len(val_loader), mse_loss / len(val_loader), l1_loss / len(val_loader)

def train_autoencoder():
    """Main training function"""
    print("🚀 Starting ECG Autoencoder Training")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    signals, ages = load_ecg_data()
    age_groups = create_age_groups(ages)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        signals, age_groups, test_size=0.2, random_state=42, stratify=age_groups
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create datasets
    train_dataset = ECGAutoencoderDataset(X_train)
    val_dataset = ECGAutoencoderDataset(X_val)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False
    )
    
    # Create model
    model = ECGAutoencoder(
        input_length=SIGNAL_LENGTH,
        latent_dim=LATENT_DIM,
        encoder_channels=ENCODER_CHANNELS,
        decoder_channels=DECODER_CHANNELS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n🎯 Training for {EPOCHS} epochs...")
    start_time = time.time()
    
    try:
        for epoch in range(EPOCHS):
            # Train
            train_loss, train_mse, train_l1 = train_epoch(
                model, train_loader, optimizer, None, DEVICE, epoch + 1
            )
            
            # Validate
            val_loss, val_mse, val_l1 = validate_epoch(
                model, val_loader, None, DEVICE
            )
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Store losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Print progress
            print(f"Epoch {epoch + 1:2d}/{EPOCHS}: "
                  f"Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, L1: {train_l1:.4f}) | "
                  f"Val Loss: {val_loss:.4f} (MSE: {val_mse:.4f}, L1: {val_l1:.4f})")
            
            # Early stopping
            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_autoencoder.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        # Cleanup
        if DEVICE.type == 'mps':
            torch.mps.empty_cache()
    
    training_time = time.time() - start_time
    print(f"\n⏱️  Training completed in {training_time:.2f} seconds")
    
    # Load best model
    if os.path.exists(f"{OUTPUT_DIR}/best_autoencoder.pth"):
        model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_autoencoder.pth"))
        print("✅ Loaded best model")
    
    # Generate visualizations
    generate_visualizations(model, val_loader, train_losses, val_losses, age_groups)
    
    return model, train_losses, val_losses

def generate_visualizations(model, val_loader, train_losses, val_losses, age_groups):
    """Generate training and analysis visualizations"""
    print("📊 Generating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss difference
    loss_diff = np.array(val_losses) - np.array(train_losses)
    axes[0, 1].plot(loss_diff, color='green', alpha=0.7)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Overfitting Indicator (Val - Train Loss)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss Difference')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    axes[1, 0].text(0.5, 0.5, 'Learning Rate Schedule\n(ReduceLROnPlateau)', 
                   ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Handle empty loss lists
    best_val_loss = min(val_losses) if val_losses else 0.0
    final_train_loss = train_losses[-1] if train_losses else 0.0
    final_val_loss = val_losses[-1] if val_losses else 0.0
    
    summary_text = f"""Model Summary:
    
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Latent Dimension: {LATENT_DIM}
Input Length: {SIGNAL_LENGTH}
Encoder Channels: {ENCODER_CHANNELS}
Decoder Channels: {DECODER_CHANNELS}
Best Validation Loss: {best_val_loss:.4f}
Final Training Loss: {final_train_loss:.4f}
Final Validation Loss: {final_val_loss:.4f}"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    axes[1, 1].set_title('Model Summary', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/autoencoder_training_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Reconstruction examples
    model.eval()
    with torch.no_grad():
        # Get a batch of validation data
        val_batch = next(iter(val_loader))
        val_data = val_batch[0][:8].to(DEVICE)  # First 8 samples
        
        # Get reconstructions
        recon_data, latent = model(val_data)
        
        # Convert to numpy
        val_data_np = val_data.cpu().numpy()
        recon_data_np = recon_data.cpu().numpy()
        latent_np = latent.cpu().numpy()
    
    # Plot reconstruction examples
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(8):
        ax = axes[i]
        
        # Original signal
        ax.plot(val_data_np[i, 0], label='Original', alpha=0.8, linewidth=1.5)
        
        # Reconstructed signal
        ax.plot(recon_data_np[i, 0], label='Reconstructed', alpha=0.8, linewidth=1.5)
        
        # Calculate reconstruction error
        mse = np.mean((val_data_np[i, 0] - recon_data_np[i, 0]) ** 2)
        mae = np.mean(np.abs(val_data_np[i, 0] - recon_data_np[i, 0]))
        
        ax.set_title(f'Sample {i+1} (MSE: {mse:.4f}, MAE: {mae:.4f})', fontsize=10)
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('ECG Signal Reconstruction Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/autoencoder_reconstruction_examples.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Latent space visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Latent space scatter plot (first 2 dimensions)
    axes[0, 0].scatter(latent_np[:, 0], latent_np[:, 1], alpha=0.6, s=20)
    axes[0, 0].set_title('Latent Space (First 2 Dimensions)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Latent Dim 1')
    axes[0, 0].set_ylabel('Latent Dim 2')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Latent space distribution
    axes[0, 1].hist(latent_np.flatten(), bins=50, alpha=0.7, color='skyblue')
    axes[0, 1].set_title('Latent Space Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Latent Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reconstruction error distribution
    recon_errors = np.mean((val_data_np - recon_data_np) ** 2, axis=(1, 2))
    axes[1, 0].hist(recon_errors, bins=30, alpha=0.7, color='lightcoral')
    axes[1, 0].set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Mean Squared Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Latent space correlation
    latent_corr = np.corrcoef(latent_np.T)
    im = axes[1, 1].imshow(latent_corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title('Latent Space Correlation Matrix', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('Latent Dimension')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/autoencoder_latent_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizations saved to ecg_autoencoder_outputs/")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🏥 ECG Autoencoder Training")
    print("=" * 30)
    
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        sys.exit(1)
    
    # Train autoencoder
    model, train_losses, val_losses = train_autoencoder()
    
    print("\n🎉 Autoencoder training completed!")
    print(f"📁 Results saved to: {OUTPUT_DIR}/")
    print("📊 Generated files:")
    print("  • autoencoder_training_analysis.png")
    print("  • autoencoder_reconstruction_examples.png") 
    print("  • autoencoder_latent_analysis.png")
    print("  • best_autoencoder.pth")
