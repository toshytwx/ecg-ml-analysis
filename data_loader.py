"""
Data loading and preprocessing module for ECG analysis.

This module provides functionality to load ECG data from the PhysioNet Autonomic Aging dataset
and prepare it for machine learning analysis.
"""

import os
import pandas as pd
import numpy as np
import wfdb
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class ECGDataset:
    """
    A class to handle loading and preprocessing of ECG data from the Autonomic Aging dataset.
    """
    
    def __init__(self, dataset_path: str, max_records: Optional[int] = None):
        """
        Initialize the ECG dataset loader.
        
        Args:
            dataset_path: Path to the autonomic aging dataset directory
            max_records: Maximum number of records to load (None for all)
        """
        self.dataset_path = dataset_path
        self.max_records = max_records
        self.subject_info = None
        self.records_list = None
        
        # Load subject information
        self._load_subject_info()
        self._load_records_list()
    
    def _load_subject_info(self):
        """Load subject information from CSV file."""
        csv_path = os.path.join(self.dataset_path, 'subject-info.csv')
        if os.path.exists(csv_path):
            self.subject_info = pd.read_csv(csv_path)
            print(f"Loaded subject info for {len(self.subject_info)} subjects")
        else:
            print("Warning: subject-info.csv not found")
            self.subject_info = None
    
    def _load_records_list(self):
        """Load list of available records."""
        records_path = os.path.join(self.dataset_path, 'RECORDS')
        if os.path.exists(records_path):
            with open(records_path, 'r') as f:
                self.records_list = [line.strip() for line in f.readlines()]
            print(f"Found {len(self.records_list)} records")
        else:
            # Fallback: scan directory for .hea files
            self.records_list = []
            for file in os.listdir(self.dataset_path):
                if file.endswith('.hea'):
                    self.records_list.append(file[:-4])  # Remove .hea extension
            self.records_list.sort()
            print(f"Found {len(self.records_list)} records by scanning directory")
    
    def load_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a single ECG record.
        
        Args:
            record_id: Record identifier (e.g., '0001')
            
        Returns:
            Dictionary containing record data and metadata, or None if loading fails
        """
        try:
            record_path = os.path.join(self.dataset_path, record_id)
            
            # Load the record
            record = wfdb.rdrecord(record_path)
            
            # Get subject info if available
            subject_info = None
            if self.subject_info is not None:
                # Convert record_id to int for comparison
                record_id_int = int(record_id)
                subject_row = self.subject_info[self.subject_info['ID'] == record_id_int]
                if not subject_row.empty:
                    subject_info = subject_row.iloc[0].to_dict()
            
            return {
                'record_id': record_id,
                'signals': record.p_signal,
                'fs': record.fs,
                'signal_names': record.sig_name,
                'units': record.units,
                'subject_info': subject_info,
                'record_length': record.sig_len
            }
            
        except Exception as e:
            print(f"Error loading record {record_id}: {str(e)}")
            return None
    
    def load_multiple_records(self, record_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Load multiple ECG records.
        
        Args:
            record_ids: List of record identifiers
            
        Returns:
            List of record dictionaries
        """
        records = []
        for record_id in tqdm(record_ids, desc="Loading records"):
            record = self.load_record(record_id)
            if record is not None:
                records.append(record)
        return records
    
    def get_valid_records(self) -> List[str]:
        """
        Get list of records that have both ECG data and subject information.
        
        Returns:
            List of valid record IDs
        """
        valid_records = []
        
        if self.subject_info is None:
            print("Warning: No subject info available, returning all records")
            return self.records_list[:self.max_records] if self.max_records else self.records_list
        
        # Filter records that have subject info and are not missing critical data
        for _, row in self.subject_info.iterrows():
            record_id = str(int(row['ID'])).zfill(4)  # Ensure 4-digit format
            
            # Skip records with missing age group
            if pd.isna(row['Age_group']):
                continue
                
            # Check if record files exist
            record_path = os.path.join(self.dataset_path, f"{record_id}.hea")
            if os.path.exists(record_path):
                valid_records.append(record_id)
        
        if self.max_records:
            valid_records = valid_records[:self.max_records]
            
        print(f"Found {len(valid_records)} valid records with complete data")
        return valid_records
    
    def get_age_groups(self) -> Dict[int, str]:
        """
        Get mapping of age group codes to descriptions.
        
        Returns:
            Dictionary mapping age group codes to descriptions
        """
        # Based on the dataset documentation, age groups are:
        age_groups = {
            1: "18-25 years",
            2: "26-35 years", 
            3: "36-45 years",
            4: "46-55 years",
            5: "56-65 years",
            6: "66-75 years",
            7: "76-85 years",
            8: "86-95 years",
            9: "96-105 years",
            10: "106-115 years",
            11: "116-125 years",
            12: "126-135 years",
            13: "136-145 years",
            14: "146-155 years",
            15: "156-165 years"
        }
        return age_groups
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the dataset.
        
        Returns:
            Dictionary containing dataset summary
        """
        if self.subject_info is None:
            return {"error": "No subject information available"}
        
        summary = {
            "total_subjects": len(self.subject_info),
            "age_groups": self.subject_info['Age_group'].value_counts().to_dict(),
            "sex_distribution": self.subject_info['Sex'].value_counts().to_dict(),
            "bmi_stats": {
                "mean": self.subject_info['BMI'].mean(),
                "std": self.subject_info['BMI'].std(),
                "min": self.subject_info['BMI'].min(),
                "max": self.subject_info['BMI'].max()
            },
            "recording_length_stats": {
                "mean": self.subject_info['Length'].mean(),
                "std": self.subject_info['Length'].std(),
                "min": self.subject_info['Length'].min(),
                "max": self.subject_info['Length'].max()
            },
            "device_distribution": self.subject_info['Device'].value_counts().to_dict()
        }
        
        return summary
    
    def load_features(self, feature_extractor=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ECG features for machine learning.
        
        Args:
            feature_extractor: Optional feature extractor instance
            
        Returns:
            Tuple of (features, labels) for machine learning
        """
        from feature_extractor import ECGFeatureExtractor
        
        if feature_extractor is None:
            feature_extractor = ECGFeatureExtractor()
        
        valid_records = self.get_valid_records()
        records = self.load_multiple_records(valid_records)
        
        features_list = []
        labels_list = []
        
        for record in tqdm(records, desc="Extracting features"):
            if record['subject_info'] is None:
                continue
                
            # Extract features
            features = feature_extractor.extract_features(record)
            if features is not None:
                features_list.append(features)
                labels_list.append(record['subject_info']['Age_group'])
        
        if not features_list:
            raise ValueError("No valid features extracted")
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each")
        return X, y
