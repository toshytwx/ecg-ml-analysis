# ECG ML Analysis - Autonomic Aging Dataset

This repository contains machine learning models for analyzing ECG data from the PhysioNet Autonomic Aging dataset to quantify changes in cardiovascular autonomic function during healthy aging.

## Dataset

The dataset contains ECG and continuous non-invasive blood pressure signals from 1,121 healthy volunteers, collected at Jena University Hospital. The data is stored in WFDB format and includes:

- **ECG signals**: 2 channels (ECG1, ECG2) at 1000 Hz sampling rate
- **Blood pressure**: 1 channel (NIBP) at 1000 Hz sampling rate
- **Subject metadata**: Age groups, sex, BMI, recording length, device type

## Features

- **Data Loading**: Efficient loading of WFDB format ECG data using the wfdb library
- **Feature Extraction**: Comprehensive ECG feature extraction including:
  - Heart rate variability (HRV) features
  - Time-domain features
  - Frequency-domain features
  - Morphological features
- **Gradient Boosting Models**: Implementation of XGBoost, LightGBM, and Random Forest for age prediction
- **Optimized Processing**: Fast feature extraction and batch processing for large datasets

## Available Scripts

- **`fast_main.py`**: Fast analysis script for testing and small datasets (20-100 records)
- **`full_dataset_main.py`**: Optimized script for training on the complete dataset (1000+ records)
- **Core modules**: `data_loader.py`, `feature_extractor.py`, `models.py`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

**Fast Analysis (Recommended for testing):**
```bash
python3 fast_main.py --max_records 50
```

**Full Dataset Training:**
```bash
python3 full_dataset_main.py
```

### Python API

```python
from data_loader import ECGDataset
from models import GradientBoostingAgePredictor

# Load dataset
dataset = ECGDataset('/path/to/autonomic-aging-dataset')
X, y = dataset.load_features()

# Train model
model = GradientBoostingAgePredictor()
model.train(X, y)

# Make predictions
predictions = model.predict(X_test)
```

## Model Performance

The gradient boosting models are designed to predict age groups based on ECG features, helping to understand how cardiovascular autonomic function changes with healthy aging.

## Citation

If you use this code, please cite the original dataset:

Schumann, A., & Bär, K. (2021). Autonomic Aging: A dataset to quantify changes of cardiovascular autonomic function during healthy aging (version 1.0.0). PhysioNet. https://doi.org/10.13026/2hsy-t491
