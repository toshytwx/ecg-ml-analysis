#!/usr/bin/env python3
"""
wfdb_parser.py

Direct parser for PhysioNet .dat/.hea files using WFDB library.
Creates a dataset that can be used directly with PyTorch for CNN training.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb
import wfdb.processing as wp
from scipy.signal import butter, filtfilt, detrend
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder
import warnings
from collections import Counter
import glob

warnings.filterwarnings('ignore')

class WFDBECGDataset(Dataset):
    """
    PyTorch Dataset for ECG signals from PhysioNet .dat/.hea files.
    Handles signal preprocessing, windowing, and augmentation directly.
    """
    
    def __init__(self, data_path, subject_info_csv, window_size_sec=10, 
                 window_step_sec=5, augment=False, preload=False):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to directory containing .dat/.hea files
            subject_info_csv: Path to CSV with subject metadata
            window_size_sec: Size of sliding windows in seconds
            window_step_sec: Step size for sliding windows in seconds
            augment: Whether to apply data augmentation
            preload: Whether to preload all data into memory
        """
        self.data_path = data_path
        self.window_size_sec = window_size_sec
        self.window_step_sec = window_step_sec
        self.augment = augment
        self.preload = preload
        
        # Load subject information
        self.subject_info = pd.read_csv(subject_info_csv)
        self.subject_info = self.subject_info.set_index('ID')
        
        # Find all .hea files
        self.hea_files = glob.glob(os.path.join(data_path, "*.hea"))
        self.hea_files.sort()
        
        print(f"Found {len(self.hea_files)} .hea files")
        
        # Process files and build dataset
        self.samples = []
        self.classes_set = set()
        self.max_channels = 0
        
        self._build_dataset()
        
        # Setup label encoder
        self.classes_ = sorted([c for c in self.classes_set if not np.isnan(c)])
        self.le = LabelEncoder()
        self.le.fit(self.classes_)
        
        print(f"Dataset created: {len(self.samples)} samples, {len(self.classes_)} classes")
        print(f"Classes: {self.classes_}")
        
        # Preload data if requested
        if self.preload:
            self._preload_data()
    
    def _build_dataset(self):
        """Build the dataset by processing all .hea files."""
        for hea_file in self.hea_files:
            try:
                record_name = os.path.splitext(os.path.basename(hea_file))[0]
                self._process_record(record_name)
            except Exception as e:
                print(f"Error processing {hea_file}: {e}")
                continue
    
    def _process_record(self, record_name):
        """Process a single record and extract windows."""
        try:
            # Read the record
            record = wfdb.rdrecord(os.path.join(self.data_path, record_name))
            signals = record.p_signal
            fs = record.fs
            
            # Get subject info
            record_id = record_name.lstrip("0")
            if record_id == "":
                record_id = "0"
            
            if record_id in self.subject_info.index.astype(str):
                row = self.subject_info.loc[self.subject_info.index.astype(str) == record_id].iloc[0]
                age_group = row["Age_group"]
                gender = row["Sex"]
                device = row["Device"]
            else:
                age_group = np.nan
                gender = np.nan
                device = np.nan
            
            # Skip if no valid age group
            if np.isnan(age_group):
                return
            
            # Update max channels
            self.max_channels = max(self.max_channels, signals.shape[1])
            self.classes_set.add(age_group)
            
            # Create windows
            window_size = int(self.window_size_sec * fs)
            window_step = int(self.window_step_sec * fs)
            
            for start in range(0, signals.shape[0] - window_size + 1, window_step):
                window = signals[start:start + window_size, :]
                
                # Store sample info
                sample_info = {
                    'record_name': record_name,
                    'start_idx': start,
                    'end_idx': start + window_size,
                    'age_group': age_group,
                    'gender': gender,
                    'device': device,
                    'fs': fs,
                    'window_size': window_size
                }
                
                if self.preload:
                    # Preprocess and store the actual data
                    processed_window = self._preprocess_window(window, fs)
                    sample_info['data'] = processed_window
                
                self.samples.append(sample_info)
                
        except Exception as e:
            print(f"Error processing record {record_name}: {e}")
    
    def _preprocess_window(self, window, fs):
        """Preprocess a window of ECG data."""
        # Sanitize signals
        processed_window = np.zeros_like(window)
        for ch in range(window.shape[1]):
            signal = self._sanitize_signal(window[:, ch], fs)
            processed_window[:, ch] = signal
        
        # Normalize each channel
        for ch in range(processed_window.shape[1]):
            signal = processed_window[:, ch]
            std = np.std(signal)
            if std > 1e-8:
                processed_window[:, ch] = (signal - np.mean(signal)) / std
            else:
                processed_window[:, ch] = signal - np.mean(signal)
        
        return processed_window
    
    def _sanitize_signal(self, signal, fs, bandpass=True):
        """Clean and filter ECG signal."""
        # Handle NaN values
        signal = np.nan_to_num(signal, nan=0.0)
        
        # Clip extreme values
        signal = np.clip(signal, -10.0, 10.0)
        
        # Skip filtering if signal is nearly constant
        if np.ptp(signal) < 1e-6:
            return signal
        
        if bandpass and fs is not None:
            # Bandpass filter 0.5-40 Hz
            nyq = 0.5 * fs
            low = 0.5 / nyq
            high = 40 / nyq
            b, a = butter(3, [low, high], btype='band')
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                signal = filtfilt(b, a, signal)
        
        # Detrend
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            signal = detrend(signal)
        
        return signal
    
    def _augment_signal(self, signal):
        """Apply data augmentation to ECG signal."""
        if not self.augment:
            return signal
        
        # Random noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.01, signal.shape)
            signal = signal + noise
        
        # Random scaling
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.8, 1.2)
            signal = signal * scale
        
        # Random time shift (circular shift)
        if np.random.random() < 0.3:
            shift = np.random.randint(-len(signal)//10, len(signal)//10)
            signal = np.roll(signal, shift, axis=0)
        
        return signal
    
    def _preload_data(self):
        """Preload all data into memory."""
        print("Preloading data...")
        for i, sample in enumerate(self.samples):
            if 'data' not in sample:
                # Load and preprocess the data
                record = wfdb.rdrecord(os.path.join(self.data_path, sample['record_name']))
                window = record.p_signal[sample['start_idx']:sample['end_idx'], :]
                sample['data'] = self._preprocess_window(window, sample['fs'])
            if i % 1000 == 0:
                print(f"Preloaded {i}/{len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.samples)} samples")
        
        sample = self.samples[idx]
        
        if self.preload and 'data' in sample:
            # Use preloaded data
            window = sample['data'].copy()
        else:
            # Load data on demand
            try:
                record = wfdb.rdrecord(os.path.join(self.data_path, sample['record_name']))
                window = record.p_signal[sample['start_idx']:sample['end_idx'], :]
                window = self._preprocess_window(window, sample['fs'])
            except Exception as e:
                print(f"Error loading record {sample['record_name']}: {e}")
                # Return a dummy sample
                window = np.zeros((sample['window_size'], self.max_channels))
        
        # Apply augmentation
        if self.augment:
            for ch in range(window.shape[1]):
                window[:, ch] = self._augment_signal(window[:, ch])
        
        # Pad channels if needed
        n_pad = self.max_channels - window.shape[1]
        if n_pad > 0:
            pad = np.zeros((window.shape[0], n_pad))
            window = np.concatenate([window, pad], axis=1)
        
        # Convert to PyTorch tensors
        # Transpose to (channels, time) for CNN
        X_tensor = torch.from_numpy(window.T).float()
        y_tensor = torch.tensor(self.le.transform([sample['age_group']])[0], dtype=torch.long)
        
        return X_tensor, y_tensor
    
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset."""
        labels = [sample['age_group'] for sample in self.samples]
        return Counter(labels)
    
    def get_sample_info(self, idx):
        """Get metadata for a specific sample."""
        return self.samples[idx]

def create_wfdb_dataset(data_path, subject_info_csv, window_size_sec=10, 
                        window_step_sec=5, augment=False, preload=False):
    """
    Create a WFDB-based ECG dataset.
    
    Args:
        data_path: Path to directory containing .dat/.hea files
        subject_info_csv: Path to CSV with subject metadata
        window_size_sec: Size of sliding windows in seconds
        window_step_sec: Step size for sliding windows in seconds
        augment: Whether to apply data augmentation
        preload: Whether to preload all data into memory
    
    Returns:
        WFDBECGDataset: PyTorch dataset for ECG signals
    """
    return WFDBECGDataset(
        data_path=data_path,
        subject_info_csv=subject_info_csv,
        window_size_sec=window_size_sec,
        window_step_sec=window_step_sec,
        augment=augment,
        preload=preload
    )

if __name__ == "__main__":
    # Example usage
    data_path = "/Users/dmytro/Diploma/ecg_ml_analysis/v2/input/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0"
    subject_info_csv = "/Users/dmytro/Diploma/ecg_ml_analysis/v2/input/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0/subject-info.csv"
    
    # Create dataset
    dataset = create_wfdb_dataset(
        data_path=data_path,
        subject_info_csv=subject_info_csv,
        window_size_sec=10,
        window_step_sec=5,
        augment=False,
        preload=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Test loading a sample
    if len(dataset) > 0:
        X, y = dataset[0]
        print(f"Sample shape: {X.shape}, Label: {y}")
        print(f"Sample info: {dataset.get_sample_info(0)}")
