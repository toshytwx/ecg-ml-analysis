"""
Machine learning models for ECG analysis.

This module provides gradient boosting models for age prediction based on ECG features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    xgb = None
    print("Warning: XGBoost not available. Install OpenMP with: brew install libomp")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False
    lgb = None
    print("Warning: LightGBM not available. Install OpenMP with: brew install libomp")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')


class GradientBoostingAgePredictor:
    """
    A comprehensive gradient boosting model for age prediction from ECG features.
    """
    
    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        """
        Initialize the age predictor.
        
        Args:
            model_type: Type of gradient boosting model ('xgboost', 'lightgbm', 'random_forest')
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        
        # Model parameters
        self.model_params = self._get_default_params()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for the selected model."""
        if self.model_type == 'xgboost':
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': -1
            }
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.random_state,
                'n_jobs': -1
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def set_params(self, params: Dict[str, Any]):
        """
        Set model parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        self.model_params.update(params)
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              test_size: float = 0.2, 
              optimize_hyperparams: bool = False,
              cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train the gradient boosting model.
        
        Args:
            X: Feature matrix
            y: Target labels (age groups)
            test_size: Fraction of data to use for testing
            optimize_hyperparams: Whether to optimize hyperparameters
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing training results
        """
        print(f"Training {self.model_type} model...")
        
        # Split data - handle small datasets without stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        except ValueError as e:
            if "least populated class" in str(e):
                print(f"Warning: Cannot stratify due to small class sizes. Using random split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state
                )
            else:
                raise e
        
        # Encode labels first (fit on all data to avoid unseen labels)
        y_encoded = self.label_encoder.fit_transform(y)
        y_train_encoded = y_encoded[:len(X_train)]
        y_test_encoded = y_encoded[len(X_train):]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            self._optimize_hyperparameters(X_train_scaled, y_train_encoded, cv_folds)
        
        # Initialize and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
        test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train_encoded, 
                                  cv=cv_folds, scoring='accuracy')
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        self.is_trained = True
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'y_test': y_test_encoded,
            'y_test_pred': y_test_pred,
            'y_test_original': y_test,
            'model_params': self.model_params
        }
        
        print(f"Training completed!")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def _create_model(self):
        """Create the model instance based on model_type."""
        if self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install OpenMP with: brew install libomp")
            return xgb.XGBClassifier(**self.model_params)
        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available. Install OpenMP with: brew install libomp")
            return lgb.LGBMClassifier(**self.model_params)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, cv_folds: int):
        """Optimize hyperparameters using grid search."""
        print("Optimizing hyperparameters...")
        
        if self.model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif self.model_type == 'lightgbm':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        model = self._create_model()
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_folds, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X, y)
        
        # Update parameters with best found
        self.model_params.update(grid_search.best_params_)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted age groups
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions_encoded = self.model.predict(X_scaled)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
        else:
            raise ValueError(f"Model {self.model_type} does not support feature importance")
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            figsize: Figure size
        """
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance - {self.model_type.upper()}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, results: Dict[str, Any], figsize: Tuple[int, int] = (10, 8)):
        """
        Plot confusion matrix.
        
        Args:
            results: Training results dictionary
            figsize: Figure size
        """
        y_test = results['y_test']
        y_pred = results['y_test_pred']
        
        # Get class names
        class_names = self.label_encoder.classes_
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {self.model_type.upper()}')
        plt.xlabel('Predicted Age Group')
        plt.ylabel('True Age Group')
        plt.tight_layout()
        plt.show()
    
    def plot_age_distribution(self, results: Dict[str, Any], figsize: Tuple[int, int] = (12, 5)):
        """
        Plot age group distribution.
        
        Args:
            results: Training results dictionary
            figsize: Figure size
        """
        y_test_original = results['y_test_original']
        y_pred_original = self.label_encoder.inverse_transform(results['y_test_pred'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # True distribution
        unique_true, counts_true = np.unique(y_test_original, return_counts=True)
        ax1.bar(unique_true, counts_true, alpha=0.7, color='skyblue')
        ax1.set_title('True Age Group Distribution')
        ax1.set_xlabel('Age Group')
        ax1.set_ylabel('Count')
        
        # Predicted distribution
        unique_pred, counts_pred = np.unique(y_pred_original, return_counts=True)
        ax2.bar(unique_pred, counts_pred, alpha=0.7, color='lightcoral')
        ax2.set_title('Predicted Age Group Distribution')
        ax2.set_xlabel('Age Group')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.model_params = model_data['model_params']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary information.
        
        Returns:
            Dictionary with model summary
        """
        if not self.is_trained:
            return {"status": "Model not trained"}
        
        summary = {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "n_classes": len(self.label_encoder.classes_) if self.label_encoder else 0,
            "class_names": self.label_encoder.classes_.tolist() if self.label_encoder else [],
            "model_params": self.model_params
        }
        
        return summary


class EnsembleAgePredictor:
    """
    Ensemble model combining multiple gradient boosting models.
    """
    
    def __init__(self, models: List[str] = None, random_state: int = 42):
        """
        Initialize ensemble predictor.
        
        Args:
            models: List of model types to include
            random_state: Random state for reproducibility
        """
        if models is None:
            models = ['random_forest']  # Default to only Random Forest
            if XGBOOST_AVAILABLE:
                models.append('xgboost')
            if LIGHTGBM_AVAILABLE:
                models.append('lightgbm')
        
        self.models = {}
        self.random_state = random_state
        
        for model_type in models:
            try:
                self.models[model_type] = GradientBoostingAgePredictor(
                    model_type=model_type, random_state=random_state
                )
            except ImportError as e:
                print(f"Warning: Could not create {model_type} model: {e}")
                continue
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train all models in the ensemble.
        
        Args:
            X: Feature matrix
            y: Target labels
            **kwargs: Additional arguments for training
            
        Returns:
            Dictionary containing results for all models
        """
        results = {}
        
        for model_type, model in self.models.items():
            print(f"\nTraining {model_type}...")
            results[model_type] = model.train(X, y, **kwargs)
        
        return results
    
    def predict(self, X: np.ndarray, method: str = 'voting') -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
            method: Ensemble method ('voting', 'averaging')
            
        Returns:
            Ensemble predictions
        """
        if method == 'voting':
            # Majority voting
            predictions = []
            for model in self.models.values():
                pred = model.predict(X)
                predictions.append(pred)
            
            # Convert to array and take mode
            predictions_array = np.array(predictions)
            ensemble_pred = []
            for i in range(predictions_array.shape[1]):
                values, counts = np.unique(predictions_array[:, i], return_counts=True)
                ensemble_pred.append(values[np.argmax(counts)])
            
            return np.array(ensemble_pred)
        
        elif method == 'averaging':
            # Average probabilities
            probabilities = []
            for model in self.models.values():
                prob = model.predict_proba(X)
                probabilities.append(prob)
            
            avg_prob = np.mean(probabilities, axis=0)
            return self.models[list(self.models.keys())[0]].label_encoder.inverse_transform(
                np.argmax(avg_prob, axis=1)
            )
        
        else:
            raise ValueError(f"Unsupported ensemble method: {method}")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """
        Get summary of all models in the ensemble.
        
        Returns:
            Dictionary with ensemble summary
        """
        summary = {}
        for model_type, model in self.models.items():
            summary[model_type] = model.get_model_summary()
        
        return summary
