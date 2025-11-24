"""
Module for probability calibration and calibration-in-the-large to target PD.

Implements:
- Platt Scaling (Logistic Calibration)
- Isotonic Regression
- Beta Calibration
- Calibration-in-the-large (adjusting intercept/slope to match target PD)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
import warnings


class PlattScaling(BaseEstimator):
    """
    Platt Scaling: fits a logistic regression on the output probabilities.
    """
    def __init__(self):
        self.lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    
    def fit(self, probas, y_true):
        """
        Fit Platt scaling.
        
        Parameters:
        -----------
        probas : array-like, shape (n_samples,)
            Predicted probabilities from base model
        y_true : array-like, shape (n_samples,)
            True binary labels
        """
        probas = np.asarray(probas).reshape(-1, 1)
        y_true = np.asarray(y_true)
        
        # Avoid log(0) and log(1)
        eps = 1e-15
        probas = np.clip(probas, eps, 1 - eps)
        
        self.lr.fit(probas, y_true)
        return self
    
    def predict_proba(self, probas):
        """
        Return calibrated probabilities.
        
        Parameters:
        -----------
        probas : array-like, shape (n_samples,)
            Predicted probabilities from base model
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Calibrated probabilities
        """
        probas = np.asarray(probas).reshape(-1, 1)
        eps = 1e-15
        probas = np.clip(probas, eps, 1 - eps)
        
        calibrated = self.lr.predict_proba(probas)[:, 1]
        return calibrated


class IsotonicCalibration(BaseEstimator):
    """
    Isotonic Regression calibration (non-parametric, monotonic).
    """
    def __init__(self):
        self.iso_reg = IsotonicRegression(out_of_bounds='clip')
    
    def fit(self, probas, y_true):
        """
        Fit isotonic regression.
        
        Parameters:
        -----------
        probas : array-like, shape (n_samples,)
            Predicted probabilities from base model
        y_true : array-like, shape (n_samples,)
            True binary labels
        """
        probas = np.asarray(probas)
        y_true = np.asarray(y_true)
        
        self.iso_reg.fit(probas, y_true)
        return self
    
    def predict_proba(self, probas):
        """
        Return calibrated probabilities.
        
        Parameters:
        -----------
        probas : array-like, shape (n_samples,)
            Predicted probabilities from base model
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Calibrated probabilities
        """
        probas = np.asarray(probas)
        calibrated = self.iso_reg.predict(probas)
        return calibrated


class BetaCalibration(BaseEstimator):
    """
    Beta Calibration using three parameters (a, b, c).
    Based on: Kull et al. "Beta calibration: a well-founded and easily implemented
    improvement on logistic calibration for binary classifiers"
    """
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None
    
    def fit(self, probas, y_true):
        """
        Fit beta calibration.
        
        Parameters:
        -----------
        probas : array-like, shape (n_samples,)
            Predicted probabilities from base model
        y_true : array-like, shape (n_samples,)
            True binary labels
        """
        probas = np.asarray(probas)
        y_true = np.asarray(y_true)
        
        # Avoid numerical issues
        eps = 1e-15
        probas = np.clip(probas, eps, 1 - eps)
        
        # Use log loss as objective
        from scipy.optimize import minimize
        
        def neg_log_likelihood(params):
            a, b, c = params
            calibrated = expit(a + b * logit(probas) + c * np.log(probas / (1 - probas)))
            calibrated = np.clip(calibrated, eps, 1 - eps)
            return -np.mean(y_true * np.log(calibrated) + (1 - y_true) * np.log(1 - calibrated))
        
        # Initial guess
        result = minimize(neg_log_likelihood, x0=[0, 1, 0], method='Nelder-Mead')
        self.a, self.b, self.c = result.x
        
        return self
    
    def predict_proba(self, probas):
        """
        Return calibrated probabilities.
        
        Parameters:
        -----------
        probas : array-like, shape (n_samples,)
            Predicted probabilities from base model
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Calibrated probabilities
        """
        probas = np.asarray(probas)
        eps = 1e-15
        probas = np.clip(probas, eps, 1 - eps)
        
        calibrated = expit(self.a + self.b * logit(probas) + self.c * np.log(probas / (1 - probas)))
        return np.clip(calibrated, eps, 1 - eps)


class CalibrationInTheLarge(BaseEstimator):
    """
    Calibration-in-the-large: adjusts intercept to match target average PD.
    
    This is a simple linear shift in log-odds space to match the desired
    average probability (e.g., 4% PD).
    """
    def __init__(self, target_pd=0.04):
        """
        Parameters:
        -----------
        target_pd : float
            Target average probability of default (e.g., 0.04 for 4%)
        """
        self.target_pd = target_pd
        self.shift = 0.0
    
    def fit(self, probas, y_true=None):
        """
        Calculate the shift needed to match target PD.
        
        Parameters:
        -----------
        probas : array-like, shape (n_samples,)
            Predicted probabilities from base model
        y_true : ignored
            Not used, present for API consistency
        """
        probas = np.asarray(probas)
        eps = 1e-15
        probas = np.clip(probas, eps, 1 - eps)
        
        # Calculate current average
        current_mean = np.mean(probas)
        
        # Calculate shift in log-odds space
        current_logit = logit(current_mean)
        target_logit = logit(self.target_pd)
        self.shift = target_logit - current_logit
        
        return self
    
    def predict_proba(self, probas):
        """
        Return calibrated probabilities with adjusted intercept.
        
        Parameters:
        -----------
        probas : array-like, shape (n_samples,)
            Predicted probabilities from base model
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Calibrated probabilities
        """
        probas = np.asarray(probas)
        eps = 1e-15
        probas = np.clip(probas, eps, 1 - eps)
        
        # Shift in log-odds space
        logits = logit(probas)
        calibrated_logits = logits + self.shift
        calibrated = expit(calibrated_logits)
        
        return np.clip(calibrated, eps, 1 - eps)


def calibrate_model_to_target_pd(model, X_cal, y_cal, X_test, target_pd=0.04, method='platt'):
    """
    Calibrate a model to target PD using a two-step approach:
    1. Apply chosen calibration method (Platt/Isotonic/Beta)
    2. Apply calibration-in-the-large to match target PD
    
    Parameters:
    -----------
    model : fitted model
        Model with predict_proba method
    X_cal : array-like
        Calibration set features
    y_cal : array-like
        Calibration set labels
    X_test : array-like
        Test set features to calibrate
    target_pd : float
        Target average probability (default 0.04 for 4%)
    method : str
        Calibration method: 'platt', 'isotonic', or 'beta'
    
    Returns:
    --------
    dict with:
        - 'calibrated_probas': calibrated probabilities on test set
        - 'calibrator': fitted calibration object
        - 'in_the_large': fitted calibration-in-the-large object
        - 'mean_pd_before': mean PD before calibration-in-the-large
        - 'mean_pd_after': mean PD after calibration-in-the-large
    """
    # Get base predictions
    probas_cal = model.predict_proba(X_cal)[:, 1] if hasattr(model.predict_proba(X_cal), 'shape') and len(model.predict_proba(X_cal).shape) > 1 else model.predict_proba(X_cal)
    probas_test = model.predict_proba(X_test)[:, 1] if hasattr(model.predict_proba(X_test), 'shape') and len(model.predict_proba(X_test).shape) > 1 else model.predict_proba(X_test)
    
    # Step 1: Apply chosen calibration method
    if method == 'platt':
        calibrator = PlattScaling()
    elif method == 'isotonic':
        calibrator = IsotonicCalibration()
    elif method == 'beta':
        calibrator = BetaCalibration()
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    calibrator.fit(probas_cal, y_cal)
    calibrated_test = calibrator.predict_proba(probas_test)
    
    mean_pd_before = np.mean(calibrated_test)
    
    # Step 2: Apply calibration-in-the-large
    citl = CalibrationInTheLarge(target_pd=target_pd)
    citl.fit(calibrated_test)
    final_calibrated = citl.predict_proba(calibrated_test)
    
    mean_pd_after = np.mean(final_calibrated)
    
    return {
        'calibrated_probas': final_calibrated,
        'calibrator': calibrator,
        'in_the_large': citl,
        'mean_pd_before': mean_pd_before,
        'mean_pd_after': mean_pd_after,
        'uncalibrated_probas': probas_test,
        'after_method_probas': calibrated_test
    }


def create_calibrated_model_pipeline(base_model, calibrator, citl):
    """
    Create a wrapper that combines base model + calibration.
    
    Parameters:
    -----------
    base_model : fitted model
    calibrator : fitted calibration object (Platt/Isotonic/Beta)
    citl : fitted CalibrationInTheLarge object
    
    Returns:
    --------
    CalibratedModelWrapper instance
    """
    return CalibratedModelWrapper(base_model, calibrator, citl)


class CalibratedModelWrapper(BaseEstimator):
    """
    Wrapper that combines base model with calibration pipeline.
    """
    def __init__(self, base_model, calibrator, citl):
        self.base_model = base_model
        self.calibrator = calibrator
        self.citl = citl
    
    def predict_proba(self, X):
        """
        Return calibrated probabilities.
        """
        # Base predictions
        probas = self.base_model.predict_proba(X)
        if hasattr(probas, 'shape') and len(probas.shape) > 1:
            probas = probas[:, 1]
        
        # Apply calibration
        probas = self.calibrator.predict_proba(probas)
        probas = self.citl.predict_proba(probas)
        
        # Return in sklearn format
        return np.column_stack([1 - probas, probas])
    
    def predict(self, X, threshold=0.5):
        """
        Return binary predictions.
        """
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)
