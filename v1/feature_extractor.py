"""
Feature extraction module for ECG analysis.

This module provides comprehensive feature extraction from ECG signals including
heart rate variability, time-domain, frequency-domain, and morphological features.
"""

import numpy as np
import scipy.signal
import scipy.stats
from scipy.fft import fft, fftfreq
from typing import Dict, Any, Optional, List
import warnings

warnings.filterwarnings('ignore')


class ECGFeatureExtractor:
    """
    A comprehensive feature extractor for ECG signals.
    """
    
    def __init__(self, fs: int = 1000):
        """
        Initialize the feature extractor.
        
        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
    
    def extract_features(self, record: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract comprehensive features from an ECG record.
        
        Args:
            record: Dictionary containing ECG record data
            
        Returns:
            Array of extracted features or None if extraction fails
        """
        try:
            signals = record['signals']
            fs = record['fs']
            
            # Use the first ECG channel (ECG1) for most features
            ecg_signal = signals[:, 0] if signals.shape[1] > 0 else None
            if ecg_signal is None:
                return None
            
            features = []
            
            # 1. Basic statistical features
            features.extend(self._extract_basic_features(ecg_signal))
            
            # 2. Heart rate variability features
            features.extend(self._extract_hrv_features(ecg_signal, fs))
            
            # 3. Frequency domain features
            features.extend(self._extract_frequency_features(ecg_signal, fs))
            
            # 4. Morphological features
            features.extend(self._extract_morphological_features(ecg_signal, fs))
            
            # 5. Wavelet features
            features.extend(self._extract_wavelet_features(ecg_signal))
            
            # 6. Non-linear features
            # TODO: figure out why freezes
            # features.extend(self._extract_nonlinear_features(ecg_signal))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
    
    def _extract_basic_features(self, signal: np.ndarray) -> List[float]:
        """Extract basic statistical features."""
        features = []
        
        # Basic statistics
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.var(signal))
        features.append(np.median(signal))
        features.append(np.min(signal))
        features.append(np.max(signal))
        features.append(np.ptp(signal))  # Peak-to-peak
        
        # Higher order moments
        features.append(scipy.stats.skew(signal))
        features.append(scipy.stats.kurtosis(signal))
        
        # Percentiles
        features.extend(np.percentile(signal, [25, 75, 90, 95, 99]))
        
        # Energy
        features.append(np.sum(signal ** 2))
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        features.append(len(zero_crossings) / len(signal))
        
        return features
    
    def _extract_hrv_features(self, signal: np.ndarray, fs: int) -> List[float]:
        """Extract heart rate variability features."""
        features = []
        
        try:
            # Detect R-peaks using simple peak detection
            peaks = self._detect_r_peaks(signal, fs)
            
            if len(peaks) < 10:  # Need minimum number of peaks
                return [0.0] * 20  # Return zeros if insufficient peaks
            
            # Calculate RR intervals
            rr_intervals = np.diff(peaks) / fs * 1000  # Convert to milliseconds
            
            # Time domain HRV features
            features.append(np.mean(rr_intervals))  # Mean RR
            features.append(np.std(rr_intervals))   # SDNN
            features.append(np.sqrt(np.mean(np.diff(rr_intervals) ** 2)))  # RMSSD
            
            # Geometric HRV features
            rr_diff = np.diff(rr_intervals)
            features.append(len(rr_diff[rr_diff > 50]) / len(rr_diff))  # pNN50
            features.append(len(rr_diff[rr_diff > 20]) / len(rr_diff))  # pNN20
            
            # Frequency domain HRV features
            freq_features = self._extract_hrv_frequency_features(rr_intervals, fs)
            features.extend(freq_features)
            
            # Additional HRV features
            features.append(np.var(rr_intervals))  # Variance of RR intervals
            features.append(scipy.stats.skew(rr_intervals))  # Skewness
            features.append(scipy.stats.kurtosis(rr_intervals))  # Kurtosis
            
            # Heart rate statistics
            heart_rates = 60000 / rr_intervals  # Convert to BPM
            features.append(np.mean(heart_rates))
            features.append(np.std(heart_rates))
            features.append(np.min(heart_rates))
            features.append(np.max(heart_rates))
            
        except Exception:
            # Return zeros if HRV extraction fails
            features = [0.0] * 20
        
        return features
    
    def _extract_hrv_frequency_features(self, rr_intervals: np.ndarray, fs: int) -> List[float]:
        """Extract frequency domain HRV features."""
        features = []
        
        try:
            # Interpolate RR intervals to regular time grid
            time_original = np.cumsum(rr_intervals) / 1000  # Convert to seconds
            time_new = np.arange(0, time_original[-1], 1/4)  # 4 Hz sampling
            
            if len(time_new) < 10:
                return [0.0] * 5
            
            rr_interp = np.interp(time_new, time_original, rr_intervals)
            
            # Remove DC component
            rr_interp = rr_interp - np.mean(rr_interp)
            
            # Compute power spectral density
            freqs, psd = scipy.signal.welch(rr_interp, fs=4, nperseg=min(256, len(rr_interp)//4))
            
            # Define frequency bands
            vlf_band = (freqs >= 0.003) & (freqs < 0.04)
            lf_band = (freqs >= 0.04) & (freqs < 0.15)
            hf_band = (freqs >= 0.15) & (freqs < 0.4)
            
            # Calculate power in each band
            vlf_power = np.trapz(psd[vlf_band], freqs[vlf_band])
            lf_power = np.trapz(psd[lf_band], freqs[lf_band])
            hf_power = np.trapz(psd[hf_band], freqs[hf_band])
            total_power = vlf_power + lf_power + hf_power
            
            features.extend([vlf_power, lf_power, hf_power, total_power])
            
            # LF/HF ratio
            if hf_power > 0:
                features.append(lf_power / hf_power)
            else:
                features.append(0.0)
                
        except Exception:
            features = [0.0] * 5
        
        return features
    
    def _extract_frequency_features(self, signal: np.ndarray, fs: int) -> List[float]:
        """Extract frequency domain features."""
        features = []
        
        try:
            # Remove DC component
            signal_centered = signal - np.mean(signal)
            
            # Compute power spectral density
            freqs, psd = scipy.signal.welch(signal_centered, fs=fs, nperseg=min(1024, len(signal)//4))
            
            # Spectral features
            features.append(np.sum(psd))  # Total power
            features.append(np.mean(psd))  # Mean power
            features.append(np.std(psd))   # Power variability
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(psd)
            features.append(freqs[dominant_freq_idx])
            
            # Spectral centroid
            features.append(np.sum(freqs * psd) / np.sum(psd))
            
            # Spectral rolloff (95% of energy)
            cumsum_psd = np.cumsum(psd)
            rolloff_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
            if len(rolloff_idx) > 0:
                features.append(freqs[rolloff_idx[0]])
            else:
                features.append(freqs[-1])
            
            # Band power ratios
            low_freq_band = (freqs >= 0.5) & (freqs < 5)
            mid_freq_band = (freqs >= 5) & (freqs < 15)
            high_freq_band = (freqs >= 15) & (freqs < 40)
            
            low_power = np.trapz(psd[low_freq_band], freqs[low_freq_band])
            mid_power = np.trapz(psd[mid_freq_band], freqs[mid_freq_band])
            high_power = np.trapz(psd[high_freq_band], freqs[high_freq_band])
            total_power = low_power + mid_power + high_power
            
            if total_power > 0:
                features.extend([low_power/total_power, mid_power/total_power, high_power/total_power])
            else:
                features.extend([0.0, 0.0, 0.0])
                
        except Exception:
            features = [0.0] * 10
        
        return features
    
    def _extract_morphological_features(self, signal: np.ndarray, fs: int) -> List[float]:
        """Extract morphological features of ECG waves."""
        features = []
        
        try:
            # Detect R-peaks
            peaks = self._detect_r_peaks(signal, fs)
            
            if len(peaks) < 5:
                return [0.0] * 15
            
            # Extract features for each heartbeat
            heartbeat_features = []
            
            for i, peak in enumerate(peaks[1:-1]):  # Skip first and last peaks
                # Define heartbeat window (around R-peak)
                window_start = max(0, peak - int(0.3 * fs))  # 300ms before R-peak
                window_end = min(len(signal), peak + int(0.3 * fs))  # 300ms after R-peak
                
                heartbeat = signal[window_start:window_end]
                
                if len(heartbeat) < 10:
                    continue
                
                # Basic morphological features
                heartbeat_features.append(np.max(heartbeat) - np.min(heartbeat))  # Amplitude
                heartbeat_features.append(np.std(heartbeat))  # Variability
                heartbeat_features.append(scipy.stats.skew(heartbeat))  # Skewness
                
                # Find Q and S points (simplified)
                q_point = self._find_q_point(heartbeat, len(heartbeat)//2)
                s_point = self._find_s_point(heartbeat, len(heartbeat)//2)
                
                if q_point is not None and s_point is not None:
                    qrs_duration = (s_point - q_point) / fs * 1000  # Convert to ms
                    heartbeat_features.append(qrs_duration)
                else:
                    heartbeat_features.append(0.0)
            
            if heartbeat_features:
                # Aggregate heartbeat features
                heartbeat_features = np.array(heartbeat_features).reshape(-1, 4)
                features.extend(np.mean(heartbeat_features, axis=0))  # Mean
                features.extend(np.std(heartbeat_features, axis=0))   # Std
                features.extend(np.median(heartbeat_features, axis=0))  # Median
            else:
                features = [0.0] * 12
                
        except Exception:
            features = [0.0] * 12
        
        return features
    
    def _extract_wavelet_features(self, signal: np.ndarray) -> List[float]:
        """Extract wavelet-based features."""
        features = []
        
        try:
            # Simple wavelet-like features using different scales
            scales = [2, 4, 8, 16, 32]
            
            for scale in scales:
                # Downsample signal
                if len(signal) >= scale:
                    downsampled = signal[::scale]
                    
                    # Extract features from downsampled signal
                    features.append(np.mean(downsampled))
                    features.append(np.std(downsampled))
                    features.append(np.var(downsampled))
                else:
                    features.extend([0.0, 0.0, 0.0])
                    
        except Exception:
            features = [0.0] * 15
        
        return features
    
    def _extract_nonlinear_features(self, signal: np.ndarray) -> List[float]:
        """Extract non-linear features."""
        features = []
        
        try:
            # Sample entropy (simplified)
            features.append(self._sample_entropy(signal, m=2, r=0.2*np.std(signal)))
            
            # Detrended fluctuation analysis (simplified)
            features.append(self._dfa_alpha(signal))
            
            # Approximate entropy
            features.append(self._approximate_entropy(signal, m=2, r=0.2*np.std(signal)))
            
        except Exception:
            features = [0.0, 0.0, 0.0]
        
        return features
    
    def _detect_r_peaks(self, signal: np.ndarray, fs: int) -> np.ndarray:
        """Simple R-peak detection algorithm."""
        try:
            # Apply bandpass filter
            nyquist = fs / 2
            low = 5 / nyquist
            high = 15 / nyquist
            b, a = scipy.signal.butter(4, [low, high], btype='band')
            filtered_signal = scipy.signal.filtfilt(b, a, signal)
            
            # Find peaks
            min_distance = int(0.3 * fs)  # Minimum 300ms between peaks
            peaks, _ = scipy.signal.find_peaks(filtered_signal, 
                                             distance=min_distance,
                                             height=np.std(filtered_signal))
            
            return peaks
            
        except Exception:
            return np.array([])
    
    def _find_q_point(self, heartbeat: np.ndarray, r_peak_idx: int) -> Optional[int]:
        """Find Q point in heartbeat."""
        try:
            # Look for minimum before R-peak
            search_start = max(0, r_peak_idx - 50)
            search_end = r_peak_idx
            q_idx = np.argmin(heartbeat[search_start:search_end]) + search_start
            return q_idx
        except:
            return None
    
    def _find_s_point(self, heartbeat: np.ndarray, r_peak_idx: int) -> Optional[int]:
        """Find S point in heartbeat."""
        try:
            # Look for minimum after R-peak
            search_start = r_peak_idx
            search_end = min(len(heartbeat), r_peak_idx + 50)
            s_idx = np.argmin(heartbeat[search_start:search_end]) + search_start
            return s_idx
        except:
            return None
    
    def _sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy (simplified version)."""
        try:
            N = len(signal)
            if N < m + 1:
                return 0.0
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _approximate_matches(m):
                patterns = np.array([signal[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                for i in range(N - m + 1):
                    template_i = patterns[i]
                    for j in range(N - m + 1):
                        if _maxdist(template_i, patterns[j], m) <= r:
                            C[i] += 1.0
                return C
            
            C = _approximate_matches(m)
            C = C / (N - m + 1.0)
            phi = np.mean(np.log(C))
            
            C = _approximate_matches(m + 1)
            C = C / (N - m)
            phi1 = np.mean(np.log(C))
            
            return phi - phi1
            
        except:
            return 0.0
    
    def _dfa_alpha(self, signal: np.ndarray) -> float:
        """Calculate DFA alpha (simplified version)."""
        try:
            N = len(signal)
            if N < 100:
                return 0.0
            
            # Integrate signal
            y = np.cumsum(signal - np.mean(signal))
            
            # Calculate fluctuation for different scales
            scales = np.logspace(1, np.log10(N//4), 10).astype(int)
            fluctuations = []
            
            for scale in scales:
                # Divide into segments
                segments = N // scale
                if segments < 2:
                    continue
                
                # Detrend each segment
                detrended = []
                for i in range(segments):
                    start = i * scale
                    end = start + scale
                    segment = y[start:end]
                    
                    # Linear detrending
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    detrended.extend(segment - trend)
                
                # Calculate fluctuation
                fluctuation = np.sqrt(np.mean(np.array(detrended) ** 2))
                fluctuations.append(fluctuation)
            
            if len(fluctuations) < 3:
                return 0.0
            
            # Fit line to log-log plot
            log_scales = np.log10(scales[:len(fluctuations)])
            log_fluctuations = np.log10(fluctuations)
            
            if len(log_scales) > 1:
                coeffs = np.polyfit(log_scales, log_fluctuations, 1)
                return coeffs[0]  # Slope (alpha)
            else:
                return 0.0
                
        except:
            return 0.0
    
    def _approximate_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate approximate entropy (simplified version)."""
        try:
            N = len(signal)
            if N < m + 1:
                return 0.0
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _approximate_matches(m):
                patterns = np.array([signal[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                for i in range(N - m + 1):
                    template_i = patterns[i]
                    for j in range(N - m + 1):
                        if _maxdist(template_i, patterns[j], m) <= r:
                            C[i] += 1.0
                return C / (N - m + 1.0)
            
            C = _approximate_matches(m)
            phi = np.mean(np.log(C))
            
            C = _approximate_matches(m + 1)
            phi1 = np.mean(np.log(C))
            
            return phi - phi1
            
        except:
            return 0.0


