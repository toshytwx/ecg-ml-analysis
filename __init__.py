"""
ECG ML Analysis Package

A machine learning package for analyzing ECG data from the PhysioNet Autonomic Aging dataset.
"""

__version__ = "1.0.0"
__author__ = "ECG ML Analysis Team"

from .data_loader import ECGDataset
from .feature_extractor import ECGFeatureExtractor
from .models import GradientBoostingAgePredictor

__all__ = [
    "ECGDataset",
    "ECGFeatureExtractor", 
    "GradientBoostingAgePredictor"
]
