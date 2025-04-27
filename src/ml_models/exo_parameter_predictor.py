"""
Machine learning models for predicting optimal exoskeleton parameters.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import copy


class ExoParameterPredictor:
    """
    Machine learning model for predicting optimal exoskeleton parameters
    based on individual user characteristics.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the predictor with a specified model type.
        
        Args:
            model_type (str): Type of ML model to use ('random_forest', 'neural_network')
        """
        self.model_type = model_type
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = None
        self.target_names = None
    
    def _create_model(self):
        """
        Create the underlying ML model based on the specified type.
        
        Returns:
            object: Initialized ML model
        """
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42
            )
        elif self.model_type == 'neural_network':
            return MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, features, targets, hyperparameter_tuning=False):
        """
        Train the model on provided features and targets.
        
        Args:
            features (DataFrame): User features
            targets (DataFrame): Optimal exoskeleton parameters
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            
        Returns:
            dict: Training results and metrics
        """
        # Store feature and target names
        self.feature_names = features.columns.tolist()
        self.target_names = targets.columns.tolist()
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Scale the data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        
        if hyperparameter_tuning:
            # Perform hyperparameter search
            self.model = self._hyperparameter_tuning(X_train_scaled, y_train_scaled)
        else:
            # Create and train the model
            self.model = self._create_model()
        
        # Train the model
        self.model.fit(X_train_scaled, y_train_scaled)
        
        # Make predictions on validation set
        y_pred_scaled = self.model.predict(X_val_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # Per-parameter metrics
        param_metrics = {}
        for i, param in enumerate(self.target_names):
            param_mse = mean_squared_error(y_val.iloc[:, i], y_pred[:, i])
            param_r2 = r2_score(y_val.iloc[:, i], y_pred[:, i])
            param_metrics[param] = {'mse': param_mse, 'r2': param_r2}
        
        return {
            'overall_mse': mse,
            'overall_r2': r2,
            'parameter_metrics': param_metrics
        }
    
    def _hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning for the selected model.
        
        Args:
            X_train (array): Training features
            y_train (array): Training targets
            
        Returns:
            object: Tuned model
        """
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestRegressor(random_state=42)
        
        elif self.model_type == 'neural_network':
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
            base_model = MLPRegressor(random_state=42, max_iter=500)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring='neg_mean_squared_error'
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def predict(self, features):
        """
        Predict optimal exoskeleton parameters for new users.
        
        Args:
            features (DataFrame): User features
            
        Returns:
            DataFrame: Predicted exoskeleton parameters
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Ensure features have the same columns as training data
        if not all(feat in features.columns for feat in self.feature_names):
            missing = set(self.feature_names) - set(features.columns)
            raise ValueError(f"Missing features: {missing}")
        
        # Reorder columns to match training data
        features = features[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler_X.transform(features)
        
        # Make predictions
        predictions_scaled = self.model.predict(features_scaled)
        
        # Inverse transform predictions
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        # Convert to DataFrame
        return pd.DataFrame(predictions, columns=self.target_names, index=features.index)
    
    def get_feature_importance(self):
        """
        Get feature importance for the trained model.
        
        Returns:
            DataFrame: Feature importance
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if self.model_type == 'random_forest':
            importances = self.model.feature_importances_
            result = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            return result
        else:
            return pd.DataFrame({'Feature': self.feature_names})
    
    def save_model(self, model_dir='models'):
        """
        Save the trained model and scalers to disk.
        
        Args:
            model_dir (str): Directory to save model files
            
        Returns:
            bool: Success status
        """
        if self.model is None:
            print("No trained model to save")
            return False
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and scalers
        joblib.dump(self.model, os.path.join(model_dir, 'exo_predictor_model.pkl'))
        joblib.dump(self.scaler_X, os.path.join(model_dir, 'feature_scaler.pkl'))
        joblib.dump(self.scaler_y, os.path.join(model_dir, 'target_scaler.pkl'))
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
        joblib.dump(metadata, os.path.join(model_dir, 'model_metadata.pkl'))
        
        return True
    
    def load_model(self, model_dir='models'):
        """
        Load a trained model from disk.
        
        Args:
            model_dir (str): Directory containing model files
            
        Returns:
            bool: Success status
        """
        try:
            # Load model and scalers
            self.model = joblib.load(os.path.join(model_dir, 'exo_predictor_model.pkl'))
            self.scaler_X = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
            self.scaler_y = joblib.load(os.path.join(model_dir, 'target_scaler.pkl'))
            
            # Load metadata
            metadata = joblib.load(os.path.join(model_dir, 'model_metadata.pkl'))
            self.model_type = metadata['model_type']
            self.feature_names = metadata['feature_names']
            self.target_names = metadata['target_names']
            
            return True
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class TransferLearningPredictor(ExoParameterPredictor):
    """
    Transfer learning model that adapts a pre-trained exoskeleton parameter 
    predictor to new users with limited data.
    """
    
    def __init__(self, base_model_path=None):
        """
        Initialize the transfer learning predictor.
        
        Args:
            base_model_path (str, optional): Path to a pre-trained model
        """
        super().__init__()
        self.base_model = None
        if base_model_path:
            self.load_base_model(base_model_path)
    
    def load_base_model(self, model_dir):
        """
        Load a pre-trained model to use as the base for transfer learning.
        
        Args:
            model_dir (str): Directory containing base model files
            
        Returns:
            bool: Success status
        """
        try:
            # Create a new predictor and load the base model
            base_predictor = ExoParameterPredictor()
            if base_predictor.load_model(model_dir):
                self.base_model = base_predictor.model
                self.scaler_X = base_predictor.scaler_X
                self.scaler_y = base_predictor.scaler_y
                self.feature_names = base_predictor.feature_names
                self.target_names = base_predictor.target_names
                self.model_type = base_predictor.model_type
                return True
            return False
        
        except Exception as e:
            print(f"Error loading base model: {e}")
            return False
    
    def adapt_to_user(self, user_features, user_targets, adaptation_strategy='fine_tune'):
        """
        Adapt the base model to a specific user with limited data.
        
        Args:
            user_features (DataFrame): User-specific features
            user_targets (DataFrame): User-specific targets
            adaptation_strategy (str): Strategy to use ('fine_tune', 'feature_adapt')
            
        Returns:
            dict: Adaptation results and metrics
        """
        if self.base_model is None:
            raise ValueError("Base model not loaded")
        
        # Scale user data
        X_user_scaled = self.scaler_X.transform(user_features)
        y_user_scaled = self.scaler_y.transform(user_targets)
        
        if adaptation_strategy == 'fine_tune':
            # Fine-tune the existing model with user data
            self.model = self._fine_tune_model(X_user_scaled, y_user_scaled)
        
        elif adaptation_strategy == 'feature_adapt':
            # Train a small adaptation layer on top of base model features
            self.model = self._feature_adaptation(X_user_scaled, y_user_scaled)
        
        else:
            raise ValueError(f"Unknown adaptation strategy: {adaptation_strategy}")
        
        # Check if we have enough samples to split for validation
        if len(user_features) > 1:
            # Evaluate on user data using train-test split
            X_train, X_val, y_train, y_val = train_test_split(
                user_features, user_targets, test_size=0.3, random_state=42
            )
        else:
            # If only one sample, use it for both training and validation
            X_val = user_features
            y_val = user_targets
        
        # Scale validation data
        X_val_scaled = self.scaler_X.transform(X_val)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_val_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        return {
            'adaptation_strategy': adaptation_strategy,
            'mse': mse,
            'r2': r2,
            'sample_size': len(user_features)
        }
    
    def _fine_tune_model(self, X, y):
        """
        Fine-tune the base model with user-specific data.
        
        Args:
            X (array): User features
            y (array): User targets
            
        Returns:
            object: Fine-tuned model
        """
        # Create a copy of the base model
        model_copy = copy.deepcopy(self.base_model)
        
        # Adjust learning rate or other parameters for fine-tuning
        if self.model_type == 'neural_network':
            # For neural networks, use a smaller learning rate
            model_copy.set_params(learning_rate_init=0.001)
        
        # Fine-tune on user data
        model_copy.fit(X, y)
        
        return model_copy
    
    def _feature_adaptation(self, X, y):
        """
        Create a small adaptation layer on top of base model features.
        This is a simple implementation - a more sophisticated approach
        would use the base model to extract features.
        
        Args:
            X (array): User features
            y (array): User targets
            
        Returns:
            object: Adapted model
        """
        # For random forest, we can create a smaller forest
        if self.model_type == 'random_forest':
            adapter = RandomForestRegressor(
                n_estimators=20,  # Smaller number of trees
                max_depth=10,
                random_state=42
            )
            
        # For neural network, we could create a smaller network
        elif self.model_type == 'neural_network':
            adapter = MLPRegressor(
                hidden_layer_sizes=(20,),
                activation='relu',
                solver='adam',
                max_iter=200,
                random_state=42
            )
        
        # Train the adapter
        adapter.fit(X, y)
        
        return adapter 