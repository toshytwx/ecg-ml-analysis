# ECG ML Analysis Project

A comprehensive machine learning project for ECG (Electrocardiogram) signal analysis and age group classification using multiple deep learning and traditional ML approaches.

## 🎯 Project Overview

This project implements and compares various machine learning models for ECG signal analysis, focusing on age group classification using the Autonomic Aging Dataset. The project includes both traditional machine learning approaches and advanced deep learning architectures.

## 📊 Dataset

- **Source**: [Autonomic Aging Dataset](https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/) - A dataset to quantify changes of cardiovascular autonomic function during healthy aging
- **Format**: WFDB (.dat/.hea files)
- **Size**: 1,121 ECG recordings
- **Classes**: 15 age groups
- **Features**: Heart rate variability, frequency domain features, time domain features
- **Citation**: Schumann, A., & Bär, K. (2021). Autonomic Aging: A dataset to quantify changes of cardiovascular autonomic function during healthy aging (version 1.0.0). _PhysioNet_. RRID:SCR_007345. <https://doi.org/10.13026/2hsy-t491>

## 🏗️ Architecture

### Deep Learning Models

1. **Autoencoder** (`01_autoencoder_training.ipynb`)
   - 1D Convolutional Autoencoder for ECG denoising
   - Feature learning and data augmentation
   - Reconstruction quality analysis

2. **CNN** (`02_cnn_training.ipynb`)
   - 1D Convolutional Neural Network
   - Residual blocks and attention mechanisms
   - Advanced regularization techniques

3. **RNN/LSTM** (`06_rnn_training.ipynb`)
   - Bidirectional LSTM with attention
   - Focal loss for class imbalance
   - Gradient accumulation and clipping

4. **Hybrid RNN + Autoencoder** (`03_hybrid_rnn_autoencoder_training.ipynb`)
   - Pre-trained autoencoder for feature extraction
   - RNN for temporal sequence modeling
   - Combined architecture for classification

### Traditional ML Models

5. **Logistic Regression** (`04_linear_training.ipynb`)
   - Feature selection with SelectKBest
   - SMOTE for class imbalance
   - GridSearchCV hyperparameter tuning

6. **Random Forest** (`05_random_forest_training.ipynb`)
   - Enhanced feature engineering
   - Ensemble methods with VotingClassifier
   - Clinically meaningful age binning

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy matplotlib seaborn
pip install wfdb tqdm imbalanced-learn
pip install jupyter notebook
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ecg_ml_analysis/v2
```

2. **Download the dataset**
   - Place the Autonomic Aging Dataset in the `input/` directory
   - Ensure the structure matches: `input/autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0/`

3. **Run the notebooks**
```bash
jupyter notebook notebooks/
```

## 📁 Project Structure

```
v2/
├── notebooks/                          # Jupyter notebooks for each model
│   ├── 01_autoencoder_training.ipynb   # Autoencoder training
│   ├── 02_cnn_training.ipynb          # CNN training
│   ├── 03_hybrid_rnn_autoencoder_training.ipynb  # Hybrid model
│   ├── 04_linear_training.ipynb       # Logistic regression
│   ├── 05_random_forest_training.ipynb # Random forest
│   └── 06_rnn_training.ipynb         # RNN/LSTM training
├── input/                             # Dataset (ignored by git)
├── output/                            # Generated features
│   └── ecg_features.csv              # Extracted features
├── *_outputs/                         # Model outputs and plots
├── train_*.py                        # Original Python training scripts
├── wfdb_parser.py                    # WFDB data parser
├── process_raw_data.py              # Data preprocessing
└── classify_with_autoencoder.py     # Autoencoder classification
```

## 🔧 Key Features

### Advanced Techniques

- **Class Imbalance Handling**
  - Focal Loss
  - SMOTE oversampling
  - Class weighting
  - Weighted random sampling

- **Regularization**
  - Dropout and Batch Normalization
  - Layer Normalization
  - Gradient clipping
  - Weight decay

- **Hyperparameter Optimization**
  - GridSearchCV
  - Learning rate scheduling
  - Early stopping
  - Cross-validation

### Comprehensive Evaluation

- **Metrics**: Accuracy, F1-score (macro/weighted), Confusion Matrix
- **Visualizations**: Training curves, feature importance, per-class performance
- **Analysis**: Class distribution, prediction confidence, model comparison

## 📈 Model Performance

Each notebook generates comprehensive outputs including:

- **Training Analysis**: Loss/accuracy curves, learning curves
- **Confusion Matrices**: Detailed classification results
- **Feature Analysis**: Importance rankings, correlation heatmaps
- **Model Summaries**: Architecture details, parameter counts
- **Per-class Performance**: Individual class accuracy and F1-scores

## 🛠️ Usage

### Running Individual Models

1. **Autoencoder Training**
```bash
jupyter notebook notebooks/01_autoencoder_training.ipynb
```

2. **CNN Training**
```bash
jupyter notebook notebooks/02_cnn_training.ipynb
```

3. **RNN Training**
```bash
jupyter notebook notebooks/06_rnn_training.ipynb
```

### Data Preprocessing

```bash
python process_raw_data.py
```

### Feature Extraction

The project includes automated feature extraction from raw ECG signals:
- Heart rate variability features
- Frequency domain analysis (LF, HF, LF/HF ratio)
- Time domain features
- Statistical measures

## 📊 Output Files

Each model generates:

- **Model files**: `*.pth`, `*.pkl`, `*.joblib`
- **Results**: JSON files with metrics and parameters
- **Visualizations**: PNG plots for analysis
- **Reports**: Classification reports and confusion matrices

## 🔬 Research Applications

This project is suitable for:

- **Cardiovascular Research**: Age-related changes in autonomic function
- **Machine Learning Research**: Comparing deep learning vs traditional ML
- **Signal Processing**: ECG denoising and feature extraction
- **Medical AI**: Automated age group classification from ECG

## 📚 Technical Details

### Deep Learning Stack
- **PyTorch**: Model implementation and training
- **MPS Support**: Apple Silicon GPU acceleration
- **Mixed Precision**: Optimized training on M4 Pro

### Traditional ML Stack
- **Scikit-learn**: Preprocessing, models, evaluation
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

### Data Processing
- **WFDB**: Physiological signal reading
- **Custom Parser**: Optimized for ECG data
- **Feature Engineering**: Domain-specific features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for research and educational purposes. Please cite appropriately if used in academic work.

## 📚 Citations

### Dataset Citation
If you use this project, please cite the original dataset:

```
Schumann, A., & Bär, K. (2021). Autonomic Aging: A dataset to quantify changes of cardiovascular autonomic function during healthy aging (version 1.0.0). _PhysioNet_. RRID:SCR_007345. <https://doi.org/10.13026/2hsy-t491>
```

### PhysioNet Platform Citation
```
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
```

## 🙏 Acknowledgments

- **Dataset**: [Autonomic Aging Dataset](https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/) by Schumann & Bär (2021)
- **Platform**: [PhysioNet](https://physionet.org/) for providing the dataset
- **Libraries**: PyTorch, Scikit-learn, WFDB
- **Community**: Open source ML and signal processing tools

## 📞 Contact

For questions or collaboration, please open an issue or contact the maintainers.

---

**Note**: This project requires the Autonomic Aging Dataset to be placed in the `input/` directory. The dataset is not included in this repository due to size constraints.
