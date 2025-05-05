"""
Model building, training and evaluation module for RiskLens Pro
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    confusion_matrix, roc_curve, precision_recall_curve
from sklearn.inspection import permutation_importance
from lime import lime_tabular
import time
from typing import Dict, List, Tuple, Any, Union, Optional

class ModelBuilder:
    """
    Class for building, training, and evaluating machine learning models
    """
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.feature_names = None
        self.lime_explainer = None
    
    def build_pipeline(self, model_name, preprocessor):
        """
        Build a pipeline for the given model name and preprocessor
        Args:
            model_name: Name of the model to build pipeline for
            preprocessor: ColumnTransformer preprocessor
        Returns:
            Pipeline: scikit-learn pipeline
        """
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
            param_dist = {
                'classifier__C': np.logspace(-4, 4, 20),
                'classifier__solver': ['liblinear', 'saga'],
                'classifier__penalty': ['l1', 'l2']
            }
        
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42)
            param_dist = {
                'classifier__n_estimators': np.linspace(100, 1000, 10, dtype=int),
                'classifier__max_depth': [None] + list(np.linspace(5, 30, 6, dtype=int)),
                'classifier__min_samples_split': np.linspace(2, 20, 5, dtype=int),
                'classifier__min_samples_leaf': np.linspace(1, 10, 5, dtype=int),
                'classifier__max_features': ['sqrt', 'log2', None]
            }
        
        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier(random_state=42)
            param_dist = {
                'classifier__n_estimators': np.linspace(100, 500, 5, dtype=int),
                'classifier__learning_rate': np.logspace(-3, 0, 10),
                'classifier__max_depth': np.linspace(3, 10, 4, dtype=int),
                'classifier__min_samples_split': np.linspace(2, 20, 5, dtype=int),
                'classifier__subsample': np.linspace(0.5, 1.0, 6)
            }
            
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Build pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        return pipeline, param_dist
    
    def train_model(self, model_name, pipeline, param_dist, X_train, y_train, X_test, y_test, cv_folds=3, n_iter=10):
        """
        Train a model with hyperparameter tuning
        Args:
            model_name: Name of the model to train
            pipeline: scikit-learn pipeline
            param_dist: Hyperparameter distribution for tuning
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            cv_folds: Number of cross-validation folds
            n_iter: Number of iterations for RandomizedSearchCV
        Returns:
            dict: Training results
        """
        # Tune hyperparameters
        search = RandomizedSearchCV(
            pipeline, param_dist, n_iter=n_iter, cv=cv_folds, scoring='roc_auc',
            n_jobs=-1, random_state=42, verbose=0
        )
        
        # Fit model
        search.fit(X_train, y_train)
        
        # Get best model
        best_pipeline = search.best_estimator_
        
        # Save model
        self.models[model_name] = best_pipeline
        
        # Evaluate model
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        # Feature importance
        feature_names, feature_importance = self._get_feature_importance(best_pipeline, X_test, y_test)
        
        return {
            'model': best_pipeline,
            'metrics': metrics,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'feature_names': feature_names,
            'feature_importance': feature_importance,
            'best_params': search.best_params_
        }
    
    def _get_feature_importance(self, pipeline, X, y):
        """
        Get feature importance for a model
        Args:
            pipeline: Trained pipeline
            X: Features
            y: Target
        Returns:
            tuple: (feature_names, feature_importance)
        """
        # Get preprocessor
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Get feature names
        try:
            # For sklearn >= 1.0
            feature_names = preprocessor.get_feature_names_out()
        except:
            # For sklearn < 1.0 (fallback)
            feature_names = self._get_feature_names_from_preprocessor(preprocessor)
        
        # Try built-in feature importance
        model = pipeline.named_steps['classifier']
        
        try:
            if hasattr(model, 'feature_importances_'):
                # For Random Forest, Gradient Boosting, etc.
                feature_importance = model.feature_importances_
                return feature_names, feature_importance
            elif hasattr(model, 'coef_'):
                # For logistic regression, etc.
                feature_importance = np.abs(model.coef_[0])
                return feature_names, feature_importance
        except:
            pass
        
        # Fallback to permutation importance
        try:
            # Transform the data
            X_processed = preprocessor.transform(X)
            
            # Calculate permutation importance
            result = permutation_importance(model, X_processed, y, n_repeats=5, random_state=42)
            feature_importance = result.importances_mean
            
            return feature_names, feature_importance
        except:
            # Return dummy importance if all else fails
            feature_importance = np.ones(len(feature_names))
            return feature_names, feature_importance
    
    def _get_feature_names_from_preprocessor(self, preprocessor):
        """
        Utility function to get feature names from a ColumnTransformer
        """
        # Initialize an empty list to store the feature names
        feature_names = []
        
        # Iterate through each transformer in the ColumnTransformer
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'drop':
                continue
            
            # For OneHotEncoder, get the feature names for all categories
            if hasattr(transformer, 'get_feature_names_out'):
                # For sklearn >= 1.0
                try:
                    trans_feature_names = transformer.get_feature_names_out(columns)
                except:
                    trans_feature_names = [f"{name}_{i}" for i in range(transformer.transform(pd.DataFrame(columns)).shape[1])]
            elif hasattr(transformer, 'get_feature_names'):
                # For sklearn < 1.0
                try:
                    trans_feature_names = transformer.get_feature_names(columns)
                except:
                    trans_feature_names = [f"{name}_{i}" for i in range(transformer.transform(pd.DataFrame(columns)).shape[1])]
            else:
                # If the transformer doesn't have a method to get feature names,
                # use the input column names
                trans_feature_names = columns
            
            # Append the feature names to the list
            feature_names.extend(trans_feature_names)
        
        return np.array(feature_names)
    
    def train_all_models(self, X_train, y_train, X_test, y_test, preprocessor, feature_names, cv_folds=3, n_iter=10):
        """
        Train all models and compare their performance
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            preprocessor: ColumnTransformer preprocessor
            feature_names: List of feature names
            cv_folds: Number of cross-validation folds
            n_iter: Number of iterations for RandomizedSearchCV
        Returns:
            dict: Dictionary of training results for all models
        """
        # Save preprocessor and feature names
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        
        # Define models to train
        model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
        
        # Initialize results dictionary
        results = {
            'metrics': {},
            'feature_importance': {},
            'confusion_matrix': None,
            'fpr': None,
            'tpr': None,
            'precision': None,
            'recall': None
        }
        
        # Train each model
        best_auc = 0
        best_model_name = None
        
        for model_name in model_names:
            # Build pipeline
            pipeline, param_dist = self.build_pipeline(model_name, preprocessor)
            
            # Train model
            model_results = self.train_model(
                model_name, pipeline, param_dist, X_train, y_train, X_test, y_test, cv_folds, n_iter
            )
            
            # Update results
            results['metrics'][model_name] = model_results['metrics']
            results['feature_importance'][model_name] = (model_results['feature_names'], model_results['feature_importance'])
            
            # Update best model
            if model_results['metrics']['roc_auc'] > best_auc:
                best_auc = model_results['metrics']['roc_auc']
                best_model_name = model_name
                results['confusion_matrix'] = model_results['confusion_matrix']
                results['fpr'] = model_results['fpr']
                results['tpr'] = model_results['tpr']
                results['precision'] = model_results['precision']
                results['recall'] = model_results['recall']
        
        # Save best model
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        # Create LIME explainer
        self._create_lime_explainer(X_train)
        
        # Add best model info to results
        results['best_model'] = best_model_name
        
        return results
    
    def _create_lime_explainer(self, X_train):
        """
        Create a LIME explainer for the trained model
        Args:
            X_train: Training features used to create the explainer
        """
        try:
            # Transform the data
            X_processed = self.preprocessor.transform(X_train)
            
            # Get the classifier
            classifier = self.best_model.named_steps['classifier']
            
            # Create the explainer
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                X_processed,
                feature_names=self.feature_names,
                class_names=["Low Risk", "High Risk"],
                mode="classification",
                random_state=42
            )
        except Exception as e:
            print(f"Error creating LIME explainer: {str(e)}")
            self.lime_explainer = None
    
    def predict(self, X):
        """
        Make predictions with the best model
        Args:
            X: Features to predict
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.best_model is None:
            raise ValueError("No trained model available for prediction")
        
        # Make predictions
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def explain_prediction(self, X, index=None):
        """
        Explain a prediction using LIME
        Args:
            X: Features to explain
            index: Index of the specific example to explain
        Returns:
            LimeTabularExplanation: LIME explanation
        """
        if self.best_model is None or self.lime_explainer is None:
            raise ValueError("Model or explainer not available")
        
        # Get the specific instance to explain
        if index is not None:
            instance = X.iloc[index]
        else:
            instance = X.iloc[0]
        
        # Transform the instance
        instance_processed = self.preprocessor.transform([instance])[0]
        
        # Get the classifier
        classifier = self.best_model.named_steps['classifier']
        
        # Get the explanation
        explanation = self.lime_explainer.explain_instance(
            instance_processed,
            classifier.predict_proba,
            num_features=10
        )
        
        return explanation
    
    def save_models(self, project_name="risk_model"):
        """
        Save trained models and preprocessor to disk
        Args:
            project_name: Base name for the saved files
        """
        # Create directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save each model
        for model_name, model in self.models.items():
            model_path = models_dir / f"{project_name}_{model_name.lower().replace(' ', '_')}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        
        # Save preprocessor
        preprocessor_path = models_dir / f"{project_name}_preprocessor.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump(self.preprocessor, f)
        
        # Save feature names
        feature_names_path = models_dir / f"{project_name}_feature_names.pkl"
        with open(feature_names_path, "wb") as f:
            pickle.dump(self.feature_names, f)
    
    def load_models(self, project_name="risk_model", model_name=None):
        """
        Load saved models and preprocessor from disk
        Args:
            project_name: Base name for the saved files
            model_name: Specific model to load (optional)
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            # Create directory path
            models_dir = Path("models")
            
            # Load preprocessor
            preprocessor_path = models_dir / f"{project_name}_preprocessor.pkl"
            with open(preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)
            
            # Load feature names
            feature_names_path = models_dir / f"{project_name}_feature_names.pkl"
            with open(feature_names_path, "rb") as f:
                self.feature_names = pickle.load(f)
            
            # Load specific model or all models
            if model_name is not None:
                model_path = models_dir / f"{project_name}_{model_name.lower().replace(' ', '_')}.pkl"
                with open(model_path, "rb") as f:
                    self.models[model_name] = pickle.load(f)
                self.best_model = self.models[model_name]
                self.best_model_name = model_name
            else:
                # Load all available models
                for model_file in models_dir.glob(f"{project_name}_*.pkl"):
                    # Skip preprocessor and feature names files
                    if "preprocessor" in model_file.name or "feature_names" in model_file.name:
                        continue
                    
                    # Extract model name
                    model_name_part = model_file.stem.replace(f"{project_name}_", "")
                    model_name = model_name_part.replace("_", " ").title()
                    
                    # Load model
                    with open(model_file, "rb") as f:
                        self.models[model_name] = pickle.load(f)
                
                # Set best model (as the first one for now)
                if self.models:
                    self.best_model_name = list(self.models.keys())[0]
                    self.best_model = self.models[self.best_model_name]
            
            # Create LIME explainer
            # This requires training data, so it can't be loaded directly
            # We'll need to create it when training data is available
            
            return True
        
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
