"""
Module for model explainability and interpretability.

Implements:
- SHAP (SHapley Additive exPlanations) analysis
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance extraction
- Partial Dependence Plots (PDP)
- Individual Conditional Expectation (ICE) curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Optional
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not installed. Install with: pip install lime")


class SHAPExplainer:
    """
    Wrapper for SHAP explanations.
    """
    def __init__(self, model, X_background, feature_names=None):
        """
        Initialize SHAP explainer.
        
        Parameters:
        -----------
        model : fitted model
            Model with predict or predict_proba method
        X_background : array-like
            Background dataset for SHAP (typically train set sample)
        feature_names : list, optional
            Feature names for display
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names
        
        # Auto-detect feature names from DataFrame
        if feature_names is None and isinstance(X_background, pd.DataFrame):
            self.feature_names = X_background.columns.tolist()
        
        # Create appropriate explainer based on model type
        self._create_explainer()
    
    def _create_explainer(self):
        """Create appropriate SHAP explainer based on model type."""
        model_type = type(self.model).__name__
        
        # Try TreeExplainer for tree-based models
        if 'RandomForest' in model_type or 'XGBoost' in model_type or 'LightGBM' in model_type:
            try:
                # For pipeline, extract the model
                if hasattr(self.model, 'named_steps'):
                    actual_model = self.model.named_steps.get('classifier', self.model)
                else:
                    actual_model = self.model
                self.explainer = shap.TreeExplainer(actual_model)
                self.explainer_type = 'tree'
            except:
                self.explainer = shap.KernelExplainer(
                    self._predict_proba_wrapper, 
                    shap.sample(self.X_background, min(100, len(self.X_background)))
                )
                self.explainer_type = 'kernel'
        else:
            # Use KernelExplainer for other models (slower but model-agnostic)
            self.explainer = shap.KernelExplainer(
                self._predict_proba_wrapper, 
                shap.sample(self.X_background, min(100, len(self.X_background)))
            )
            self.explainer_type = 'kernel'
    
    def _predict_proba_wrapper(self, X):
        """Wrapper for predict_proba that returns probabilities for positive class."""
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)
            if len(proba.shape) > 1:
                return proba[:, 1]
            return proba
        else:
            return self.model.predict(X)
    
    def explain_instance(self, X):
        """
        Get SHAP values for instances.
        
        Parameters:
        -----------
        X : array-like
            Instances to explain
        
        Returns:
        --------
        shap_values : SHAP values
        """
        if self.explainer_type == 'tree':
            # For tree explainer, we need to transform data first if it's a pipeline
            if hasattr(self.model, 'named_steps'):
                X_transformed = self.model[:-1].transform(X)
            else:
                X_transformed = X
            shap_values = self.explainer.shap_values(X_transformed)
        else:
            shap_values = self.explainer.shap_values(X)
        
        # Handle multi-output (binary classification returns list)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        return shap_values
    
    def plot_summary(self, X, max_display=20, plot_type='dot'):
        """
        Plot SHAP summary plot.
        
        Parameters:
        -----------
        X : array-like
            Dataset to explain
        max_display : int
            Maximum number of features to display
        plot_type : str
            'dot' or 'bar'
        """
        shap_values = self.explain_instance(X)
        
        if self.explainer_type == 'tree' and hasattr(self.model, 'named_steps'):
            X_transformed = self.model[:-1].transform(X)
        else:
            X_transformed = X
        
        shap.summary_plot(shap_values, X_transformed, 
                         feature_names=self.feature_names,
                         max_display=max_display,
                         plot_type=plot_type,
                         show=False)
        plt.tight_layout()
    
    def plot_waterfall(self, X, index=0):
        """
        Plot waterfall plot for a single instance.
        
        Parameters:
        -----------
        X : array-like
            Dataset
        index : int
            Index of instance to explain
        """
        shap_values = self.explain_instance(X)
        
        if isinstance(X, pd.DataFrame):
            instance = X.iloc[index:index+1]
        else:
            instance = X[index:index+1]
        
        if self.explainer_type == 'tree' and hasattr(self.model, 'named_steps'):
            instance_transformed = self.model[:-1].transform(instance)
        else:
            instance_transformed = instance
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[index],
            base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            data=instance_transformed[0] if hasattr(instance_transformed, '__getitem__') else instance_transformed,
            feature_names=self.feature_names
        )
        
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
    
    def get_feature_importance(self, X):
        """
        Get global feature importance from SHAP values.
        
        Parameters:
        -----------
        X : array-like
            Dataset to explain
        
        Returns:
        --------
        DataFrame with feature importance
        """
        shap_values = self.explain_instance(X)
        
        # Calculate mean absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)
        
        feature_names = self.feature_names if self.feature_names else [f"Feature_{i}" for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df


class LIMEExplainer:
    """
    Wrapper for LIME explanations.
    """
    def __init__(self, model, X_train, feature_names=None, class_names=None, mode='classification'):
        """
        Initialize LIME explainer.
        
        Parameters:
        -----------
        model : fitted model
            Model with predict_proba method
        X_train : array-like
            Training data for LIME
        feature_names : list, optional
            Feature names
        class_names : list, optional
            Class names (e.g., ['No Default', 'Default'])
        mode : str
            'classification' or 'regression'
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed. Install with: pip install lime")
        
        self.model = model
        self.X_train = X_train
        self.mode = mode
        
        # Auto-detect feature names
        if feature_names is None and isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = feature_names
        
        self.class_names = class_names or ['0', '1']
        
        # Convert to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            X_train_array = X_train.values
        else:
            X_train_array = X_train
        
        # Create LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_array,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode
        )
    
    def explain_instance(self, instance, num_features=10):
        """
        Explain a single instance.
        
        Parameters:
        -----------
        instance : array-like
            Single instance to explain
        num_features : int
            Number of features to include in explanation
        
        Returns:
        --------
        LIME explanation object
        """
        if isinstance(instance, pd.Series):
            instance = instance.values
        elif isinstance(instance, pd.DataFrame):
            instance = instance.values[0]
        
        if self.mode == 'classification':
            explanation = self.explainer.explain_instance(
                instance,
                self.model.predict_proba,
                num_features=num_features
            )
        else:
            explanation = self.explainer.explain_instance(
                instance,
                self.model.predict,
                num_features=num_features
            )
        
        return explanation
    
    def plot_explanation(self, explanation, label=1):
        """
        Plot LIME explanation.
        
        Parameters:
        -----------
        explanation : LIME explanation object
        label : int
            Class label to explain (1 for positive class)
        """
        fig = explanation.as_pyplot_figure(label=label)
        plt.tight_layout()
        return fig


def get_feature_importance_from_model(model, feature_names=None):
    """
    Extract feature importance from tree-based models.
    
    Parameters:
    -----------
    model : fitted model
        Tree-based model (RandomForest, XGBoost, etc.)
    feature_names : list, optional
        Feature names
    
    Returns:
    --------
    DataFrame with feature importance
    """
    # Handle pipeline
    if hasattr(model, 'named_steps'):
        actual_model = model.named_steps.get('classifier', model)
    else:
        actual_model = model
    
    # Extract importance
    if hasattr(actual_model, 'feature_importances_'):
        importance = actual_model.feature_importances_
    elif hasattr(actual_model, 'coef_'):
        # For linear models, use absolute coefficients
        importance = np.abs(actual_model.coef_[0] if len(actual_model.coef_.shape) > 1 else actual_model.coef_)
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute")
    
    # Get feature names
    if feature_names is None:
        # Try to get from pipeline
        if hasattr(model, 'named_steps') and hasattr(model[:-1], 'get_feature_names_out'):
            feature_names = model[:-1].get_feature_names_out()
        else:
            feature_names = [f"Feature_{i}" for i in range(len(importance))]
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return df


def plot_feature_importance(importance_df, top_n=20, title='Feature Importance'):
    """
    Plot feature importance as horizontal bar chart.
    
    Parameters:
    -----------
    importance_df : DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int
        Number of top features to display
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    
    top_features = importance_df.head(top_n)
    
    ax.barh(range(len(top_features)), top_features['importance'].values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def analyze_logistic_regression_coefficients(model, feature_names=None, top_n=20):
    """
    Analyze and visualize logistic regression coefficients.
    
    Parameters:
    -----------
    model : fitted LogisticRegression
        Logistic regression model (or pipeline containing it)
    feature_names : list, optional
        Feature names
    top_n : int
        Number of top features to display
    
    Returns:
    --------
    DataFrame with coefficients and odds ratios
    """
    # Handle pipeline
    if hasattr(model, 'named_steps'):
        lr_model = model.named_steps.get('classifier', model)
        # Try to get feature names from preprocessing
        if hasattr(model[:-1], 'get_feature_names_out'):
            feature_names = model[:-1].get_feature_names_out()
    else:
        lr_model = model
    
    # Extract coefficients
    if hasattr(lr_model, 'coef_'):
        coef = lr_model.coef_[0] if len(lr_model.coef_.shape) > 1 else lr_model.coef_
        intercept = lr_model.intercept_[0] if hasattr(lr_model.intercept_, '__len__') else lr_model.intercept_
    else:
        raise ValueError("Model does not have coefficients")
    
    # Get feature names
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(coef))]
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef,
        'abs_coefficient': np.abs(coef),
        'odds_ratio': np.exp(coef),
        'log_odds': coef
    }).sort_values('abs_coefficient', ascending=False)
    
    # Add interpretation
    df['effect'] = df['coefficient'].apply(lambda x: 'Increases risk' if x > 0 else 'Decreases risk')
    
    print(f"\nLogistic Regression Analysis")
    print(f"Intercept: {intercept:.4f}")
    print(f"\nTop {top_n} features by absolute coefficient:")
    print(df.head(top_n).to_string(index=False))
    
    return df


def compare_feature_importance(importance_dict, top_n=15, title='Feature Importance Comparison'):
    """
    Compare feature importance across multiple models.
    
    Parameters:
    -----------
    importance_dict : dict
        Dictionary of {model_name: importance_df}
    top_n : int
        Number of top features to display
    title : str
        Plot title
    """
    # Get all unique features
    all_features = set()
    for df in importance_dict.values():
        all_features.update(df['feature'].tolist())
    
    # Get top features from each model
    top_features = set()
    for df in importance_dict.values():
        top_features.update(df.head(top_n)['feature'].tolist())
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({'feature': list(top_features)})
    
    for model_name, df in importance_dict.items():
        importance_map = dict(zip(df['feature'], df['importance']))
        comparison[model_name] = comparison['feature'].map(importance_map).fillna(0)
    
    # Sort by average importance
    comparison['avg_importance'] = comparison.iloc[:, 1:].mean(axis=1)
    comparison = comparison.sort_values('avg_importance', ascending=False).head(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
    
    x = np.arange(len(comparison))
    width = 0.8 / len(importance_dict)
    
    for i, model_name in enumerate(importance_dict.keys()):
        offset = (i - len(importance_dict)/2 + 0.5) * width
        ax.barh(x + offset, comparison[model_name].values, width, label=model_name)
    
    ax.set_yticks(x)
    ax.set_yticklabels(comparison['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig, ax, comparison
