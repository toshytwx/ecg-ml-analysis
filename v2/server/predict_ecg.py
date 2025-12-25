#!/usr/bin/env python3
"""
ECG Age Group Prediction Script
Processes uploaded ECG files and predicts age group using trained CNN model
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import wfdb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to Python path to import wfdb_parser
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from wfdb_parser import create_wfdb_dataset
except ImportError:
    print("Warning: wfdb_parser not found, using basic WFDB reading")

# Model architecture (copied from train_cnn.py)
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

def load_ecg_signal(hea_path, dat_path):
    """Load ECG signal from .hea and .dat files"""
    try:
        # Read the WFDB record - use the full path without extension
        record_name = os.path.splitext(hea_path)[0]
        
        # Try different methods to read the signal
        try:
            # Method 1: Standard reading
            record = wfdb.rdrecord(record_name, physical=True)
            if record.p_signal is not None:
                signal = record.p_signal[:, 0]  # Take first channel
            else:
                raise Exception("p_signal is None")
        except:
            try:
                # Method 2: Read without physical conversion
                record = wfdb.rdrecord(record_name, physical=False)
                if record.d_signal is not None:
                    signal = record.d_signal[:, 0]  # Take first channel
                else:
                    raise Exception("d_signal is None")
            except:
                # Method 3: Direct signal reading
                signal = wfdb.rdsamp(record_name)[0][:, 0]
        
        # Ensure signal is not empty
        if len(signal) == 0:
            raise Exception("Empty signal")
        
        # Normalize the signal
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        
        return signal.astype(np.float32)
    except Exception as e:
        raise Exception(f"Failed to load ECG signal: {str(e)}")

def preprocess_signal(signal, target_length=2500):
    """Preprocess ECG signal to match training format"""
    # Pad or truncate to target length
    if len(signal) > target_length:
        signal = signal[:target_length]
    else:
        signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
    
    # Reshape for CNN input (batch_size, channels, sequence_length)
    # The model expects 3 channels, so we'll duplicate the signal
    signal = signal.reshape(1, -1)
    signal = np.tile(signal, (3, 1))  # Duplicate to 3 channels
    signal = signal.reshape(1, 3, -1)  # Shape: (1, 3, length)
    
    return signal

def predict_age_group(session_dir, model_path):
    """Predict age group from ECG files"""
    try:
        # Look for .hea and .dat files in the directory
        hea_files = [f for f in os.listdir(session_dir) if f.endswith('.hea')]
        dat_files = [f for f in os.listdir(session_dir) if f.endswith('.dat')]
        
        if not hea_files or not dat_files:
            raise Exception("ECG files not found")
        
        # Use the first .hea and .dat files found
        hea_path = os.path.join(session_dir, hea_files[0])
        dat_path = os.path.join(session_dir, dat_files[0])
        
        # Check if files exist
        if not os.path.exists(hea_path) or not os.path.exists(dat_path):
            raise Exception("ECG files not found")
        
        # Load ECG signal
        signal = load_ecg_signal(hea_path, dat_path)
        
        # Preprocess signal
        processed_signal = preprocess_signal(signal)
        signal_snippet = processed_signal[0:2000].flatten().tolist() if len(processed_signal) > 0 else []
        # Convert to tensor
        signal_tensor = torch.FloatTensor(processed_signal)
        
        # Load trained model
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found: {model_path}")
        
        # Initialize model (15 classes for age groups, 3 channels)
        model = AdvancedCNN1D(in_channels=3, n_classes=15)
        
        # Load model weights
        device = torch.device('cpu')  # Use CPU for inference
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Make prediction
        with torch.no_grad():
            signal_tensor = signal_tensor.to(device)
            outputs = model(signal_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Convert probabilities to list
        prob_list = probabilities[0].cpu().numpy().tolist()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': prob_list,
            'signal_data': signal_snippet
        }
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

def main():
    if len(sys.argv) != 3:
        print(json.dumps({'error': 'Usage: python predict_ecg.py <session_dir> <model_path>'}))
        sys.exit(1)
    
    session_dir = sys.argv[1]
    model_path = sys.argv[2]
    
    try:
        result = predict_age_group(session_dir, model_path)
        print(json.dumps(result))
    except Exception as e:
        error_result = {'error': str(e)}
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == '__main__':
    main()
