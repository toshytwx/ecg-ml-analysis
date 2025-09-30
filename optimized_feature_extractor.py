import numpy as np
from typing import Dict, Any

class OptimizedFeatureExtractor:
    """
    Optimized feature extractor for full dataset training.
    """
    
    def __init__(self, fs: int = 1000):
        self.fs = fs
    
    def extract_features(self, record: Dict[str, Any]) -> np.ndarray:
        """Extract optimized features from ECG record."""
        try:
            signals = record['signals']
            if signals is None or signals.shape[1] == 0:
                return None
            
            # Use first ECG channel
            ecg_signal = signals[:, 0]
            
            features = []
            
            # 1. Basic statistical features
            features.extend([
                np.mean(ecg_signal),
                np.std(ecg_signal),
                np.var(ecg_signal),
                np.median(ecg_signal),
                np.min(ecg_signal),
                np.max(ecg_signal),
                np.ptp(ecg_signal),  # Peak-to-peak
            ])
            
            # 2. Percentiles
            features.extend(np.percentile(ecg_signal, [25, 75, 90, 95]))
            
            # 3. Heart rate estimation (optimized)
            try:
                from scipy.signal import find_peaks
                # Use a subset for peak detection to speed up
                subset_size = min(30000, len(ecg_signal))  # 30 seconds max
                signal_subset = ecg_signal[:subset_size]
                
                peaks, _ = find_peaks(signal_subset, 
                                    distance=int(0.3 * self.fs), 
                                    height=np.std(signal_subset))
                
                if len(peaks) > 5:
                    rr_intervals = np.diff(peaks) / self.fs * 1000
                    features.extend([
                        np.mean(rr_intervals),
                        np.std(rr_intervals),
                        60000 / np.mean(rr_intervals),
                        len(peaks) / (subset_size / self.fs) * 60,
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
            except:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # 4. Frequency features (on smaller subset)
            try:
                subset_size = min(2000, len(ecg_signal))  # 2 seconds max
                signal_subset = ecg_signal[:subset_size]
                signal_centered = signal_subset - np.mean(signal_subset)
                
                fft_signal = np.fft.fft(signal_centered)
                power_spectrum = np.abs(fft_signal) ** 2
                
                features.extend([
                    np.sum(power_spectrum),
                    np.mean(power_spectrum),
                    np.std(power_spectrum),
                ])
            except:
                features.extend([0.0, 0.0, 0.0])
            
            # 5. Additional features
            features.extend([
                len(ecg_signal),
                record['fs'],
                record['record_length'],
            ])
            
            return np.array(features)
            
        except Exception as e:
            return None