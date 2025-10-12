from data_loader import ECGDataset
from feature_extractor import ECGFeatureExtractor
import numpy as np
import pandas as pd
import os


def save_features(X, y, output_path='extracted_features.csv'):
    """Save features and labels to CSV file"""
    df = pd.DataFrame(X)
    df['label'] = y
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")

def load_features(input_path='extracted_features.csv'):
    """Load features and labels from CSV file"""
    if not os.path.exists(input_path):
        return None, None
    df = pd.read_csv(input_path)
    y = df['label'].values
    X = df.drop('label', axis=1).values
    print(f"Features loaded from {input_path}")
    return X, y

def main():
    features_path = 'extracted_features.csv'
    
    # Try to load existing features first
    X, y = load_features(features_path)
    if X is not None and y is not None:
        print(f"Loaded existing features:")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Number of age groups: {len(np.unique(y))}")
    else:
        # If no existing features, extract new ones
        print("No existing features found. Extracting new features...")
        dataset_path = "/Users/dmytro/Diploma/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0"
        print("\n1. Loading dataset and extracting features...")
        dataset = ECGDataset(dataset_path)
        feature_extractor = ECGFeatureExtractor()
        
        valid_records = dataset.get_valid_records()
        total_records = len(valid_records)
        features_list = []
        labels_list = []
        print(f"Processing {total_records} valid records...")
        
        for idx, record_id in enumerate(valid_records):
            print(f"Processing record {idx + 1}/{total_records}: {record_id}")
            record = dataset.load_record(record_id)
            if record is None:
                continue
            features = feature_extractor.extract_features(record)
            if features is not None:
                features_list.append(features)
                labels_list.append(record['subject_info']['Age_group'])

        if len(features_list) == 0:
            print("No features extracted!")
            return
        
        try:
            X = np.array(features_list)
        except ValueError:
            max_length = max(len(f) for f in features_list)
            X = np.full((len(features_list), max_length), np.nan)
            for i, feat in enumerate(features_list):
                feat_array = np.array(feat)
                X[i, :len(feat_array)] = feat_array
        
        y = np.array(labels_list)
    
    print(f"\nFeature extraction completed!")
    print(f"  Total records processed: {len(features_list)}")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Number of age groups: {len(np.unique(y))}")
    
    save_features(X, y, features_path)

if __name__ == "__main__":
    main()