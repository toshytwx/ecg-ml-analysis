import wfdb
import numpy as np
import pandas as pd
import os
from scipy.stats import skew, kurtosis
import glob
from sklearn.decomposition import PCA
import warnings
from scipy.signal import butter, filtfilt, welch, detrend
import wfdb.processing as wp

# ---------------- CONFIG ----------------
DATA_PATH = "/Users/dmytro/Diploma/ecg_ml_analysis/v2/input/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0"              # Folder with .dat and .hea files
SUBJECT_INFO_CSV = "/Users/dmytro/Diploma/ecg_ml_analysis/v2/input/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0/subject-info.csv"
TABULAR_CSV = "/Users/dmytro/Diploma/ecg_ml_analysis/v2/output/ecg_features.csv"
ECG_CLIP_RANGE = (-10.0, 10.0)  # Physiological ECG range in mV

# ---------------- HELPER FUNCTIONS ----------------
def safe_skew(signal):
    signal = np.nan_to_num(signal, nan=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return skew(signal)
        except Exception:
            print("Error computing skewness, falling back to NaN")
            return np.nan

def safe_kurtosis(signal):
    signal = np.nan_to_num(signal, nan=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return kurtosis(signal)
        except Exception:
            print("Error computing kurtosis, falling back to NaN")
            return np.nan

def sanitize_signal(signal, fs=None, bandpass=True):
    """Clip extreme values, replace NaNs with 0, optional bandpass 0.5-40Hz."""
    signal = np.nan_to_num(signal, nan=0.0)
    signal = np.clip(signal, ECG_CLIP_RANGE[0], ECG_CLIP_RANGE[1])

    # Skip filtering if signal is nearly constant
    if np.ptp(signal) < 1e-6:
        return signal

    if bandpass and fs is not None:
        nyq = 0.5 * fs
        low = 0.5 / nyq
        high = 40 / nyq
        b, a = butter(3, [low, high], btype='band')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            signal = filtfilt(b, a, signal)

    # Safe detrend
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        signal = detrend(signal)

    return signal

def extract_signal_features(signal, fs):
    features = {}
    signal = sanitize_signal(signal, fs=fs, bandpass=True)

    # Time-domain features
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['min'] = np.min(signal)
    features['max'] = np.max(signal)
    features['median'] = np.median(signal)
    features['skew'] = safe_skew(signal)
    features['kurtosis'] = safe_kurtosis(signal)

    # --- R-peaks and HRV using WFDB GQRS ---
    try:
        peaks = wp.gqrs_detect(sig=signal, fs=fs)
        if len(peaks) < 3:  # Need at least 3 peaks for meaningful HRV
            raise ValueError("Insufficient R-peaks detected")
        
        rr_intervals = np.diff(peaks) / fs
        # Filter out physiologically impossible RR intervals
        rr_intervals = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
        
        if len(rr_intervals) < 2:
            raise ValueError("Insufficient valid RR intervals")
            
    except Exception as e:
        print(f"R-peak detection failed: {e}")
        rr_intervals = np.array([])

    if len(rr_intervals) > 1:
        features['hr_mean'] = 60 / np.mean(rr_intervals)
        features['hr_std'] = 60 * np.std(rr_intervals)
        features['rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals)**2))
        features['nn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 0.05)
        features['pnn50'] = features['nn50'] / len(rr_intervals)

        # Frequency-domain HRV (Welch)
        rr_mean = np.mean(rr_intervals)
        rr_detrended = rr_intervals - rr_mean
        fxx, pxx = welch(rr_detrended, fs=1.0/rr_mean, nperseg=min(256, len(rr_detrended)))
        vlf_band = (0.0033, 0.04)
        lf_band  = (0.04, 0.15)
        hf_band  = (0.15, 0.4)

        def band_power(f, Pxx, band):
            mask = (f >= band[0]) & (f <= band[1])
            return np.trapezoid(Pxx[mask], f[mask]) if np.any(mask) else 0.0

        features['VLF'] = band_power(fxx, pxx, vlf_band)
        features['LF']  = band_power(fxx, pxx, lf_band)
        features['HF']  = band_power(fxx, pxx, hf_band)
        features['LFHF'] = features['LF'] / features['HF'] if features['HF'] > 0 else 0.0
    else:
        print("Not enough R-peaks detected for HRV, using zeros")
        features.update({k: 0.0 for k in ['hr_mean','hr_std','rmssd','nn50','pnn50','VLF','LF','HF','LFHF']})

    return features

def extract_multi_lead_features(signals):
    n_channels = signals.shape[1]
    features = {}
    signals_clean = np.clip(np.nan_to_num(signals, nan=0.0), ECG_CLIP_RANGE[0], ECG_CLIP_RANGE[1])

    # Correlation and covariance
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.corrcoef(signals_clean.T)
            cov_matrix = np.cov(signals_clean.T)
            upper_idx = np.triu_indices(n_channels, k=1)
            for r, c in zip(*upper_idx):
                if not (np.isnan(corr_matrix[r, c]) or np.isnan(cov_matrix[r, c])):
                    features[f'corr_ch{r}_ch{c}'] = corr_matrix[r, c]
                    features[f'cov_ch{r}_ch{c}'] = cov_matrix[r, c]
                else:
                    features[f'corr_ch{r}_ch{c}'] = 0.0
                    features[f'cov_ch{r}_ch{c}'] = 0.0
    except Exception:
        print("Error computing correlation/covariance, using zeros")
        for r in range(n_channels):
            for c in range(r+1, n_channels):
                features[f'corr_ch{r}_ch{c}'] = 0.0
                features[f'cov_ch{r}_ch{c}'] = 0.0
    # PCA
    try:
        if signals_clean.shape[0] > signals_clean.shape[1]:  # More samples than features
            pca = PCA(n_components=min(n_channels, signals_clean.shape[0]))
            pca.fit(signals_clean)
            for i, ratio in enumerate(pca.explained_variance_ratio_):
                features[f'pca_ratio_{i}'] = ratio
            # Fill remaining PCA ratios with 0
            for i in range(len(pca.explained_variance_ratio_), 3):
                features[f'pca_ratio_{i}'] = 0.0
        else:
            # Not enough samples for PCA
            for i in range(3):
                features[f'pca_ratio_{i}'] = 0.0
    except Exception:
        print("Error computing PCA, using zeros")
        for i in range(3):
            features[f'pca_ratio_{i}'] = 0.0

    return features

# ---------------- MAIN ----------------
def main():
    all_tabular_features = []

    subject_info = pd.read_csv(SUBJECT_INFO_CSV)
    subject_info = subject_info.set_index('ID')

    hea_files = glob.glob(os.path.join(DATA_PATH, "*.hea"))
    
    print(f"Processing {len(hea_files)} files (memory-efficient mode)...")

    for i, hea_file in enumerate(hea_files):
        record_name = os.path.splitext(os.path.basename(hea_file))[0]
        try:
            # Read ECG record
            record = wfdb.rdrecord(os.path.join(DATA_PATH, record_name))
            fs = record.fs
            signals = record.p_signal

            # Patient info
            record_id = record_name.lstrip("0")  # remove leading zeros
            if record_id == "":
                record_id = "0"

            print(f"Processing record: {record_name}, fs={fs}, shape={signals.shape}")
            if record_id in subject_info.index.astype(str):
                print(f"Found subject info for {record_name}")
                row = subject_info.loc[subject_info.index.astype(str) == record_id].iloc[0]
                age_group = row["Age_group"]
                gender = row["Sex"]
                device = row["Device"]
                print(f"Age group: {age_group}, gender: {gender}, device: {device}")
            else:
                age_group = np.nan
                gender = np.nan
                device = np.nan

            # Extract multi-lead features
            multi_features = extract_multi_lead_features(signals)

            # Aggregate features across channels
            channel_features = []
            for ch in range(signals.shape[1]):
                signal = signals[:, ch]
                f = extract_signal_features(signal, fs)
                channel_features.append(f)

            # Take mean of all channel features
            df_ch = pd.DataFrame(channel_features)
            features_agg = df_ch.mean().to_dict()

            # Add multi-lead features and metadata
            features_agg.update(multi_features)
            features_agg['record'] = record_name
            features_agg['age_group'] = age_group
            features_agg['gender'] = gender
            features_agg['device'] = device

            all_tabular_features.append(features_agg)

        except Exception as e:
            print(f"Error processing {record_name}: {e}")

    # Save aggregated tabular features CSV
    df = pd.DataFrame(all_tabular_features)
    df.to_csv(TABULAR_CSV, index=False)
    print(f"Tabular features saved to {TABULAR_CSV}")

if __name__ == "__main__":
    main()