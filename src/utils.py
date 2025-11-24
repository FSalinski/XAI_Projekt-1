import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    recall_score, roc_auc_score, precision_score, f1_score,
    average_precision_score, brier_score_loss, log_loss,
    confusion_matrix, classification_report, roc_curve,
    precision_recall_curve
)
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

def find_correlated_pairs(corr, threshold=0.75):
    correlated_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold and i != j:
                correlated_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    return correlated_pairs

def remove_correlated_features(df, correlated_pairs):
    features_to_remove = []
    for feature1, feature2, _ in correlated_pairs:
        if feature2 not in features_to_remove and feature1 not in features_to_remove:
            features_to_remove.append(feature2)
    return df.drop(columns=features_to_remove)

def remove_all_correlated_features(df, threshold=0.75):
    while True:
        corr = df.corr(numeric_only=True)
        correlated_pairs = find_correlated_pairs(corr, threshold)
        if not correlated_pairs: # if no more correlated pairs, break
            break
        df = remove_correlated_features(df, correlated_pairs)
    return df

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return recall, roc_auc


def evaluate_model_comprehensive(model, X_test, y_test, threshold=0.5):
    """
    Comprehensive evaluation with multiple metrics.
    
    Parameters:
    -----------
    model : fitted model with predict_proba
    X_test : test features
    y_test : true labels
    threshold : classification threshold (default 0.5)
    
    Returns:
    --------
    dict with all metrics
    """
    # Get predictions
    y_proba = model.predict_proba(X_test)
    if len(y_proba.shape) > 1:
        y_proba = y_proba[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Classification metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        # Probabilistic metrics
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
        'brier_score': brier_score_loss(y_test, y_proba),
        'log_loss': log_loss(y_test, y_proba),
        
        # Classification metrics
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        
        # Confusion matrix
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        
        # KS statistic
        'ks_statistic': calculate_ks_statistic(y_test, y_proba),
        
        # Calibration metrics
        'ece': expected_calibration_error(y_test, y_proba),
        'mean_predicted_probability': np.mean(y_proba),
        'mean_actual_rate': np.mean(y_test),
    }
    
    return metrics


def calculate_ks_statistic(y_true, y_proba):
    """
    Calculate Kolmogorov-Smirnov statistic.
    Measures maximum separation between cumulative distributions of scores for positive and negative classes.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities
    
    Returns:
    --------
    float : KS statistic (0 to 1, higher is better)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    # Separate scores by class
    scores_pos = y_proba[y_true == 1]
    scores_neg = y_proba[y_true == 0]
    
    # KS test
    ks_stat, _ = ks_2samp(scores_pos, scores_neg)
    
    return ks_stat


def expected_calibration_error(y_true, y_proba, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted probabilities and actual frequencies
    across probability bins.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for probability bucketing
    
    Returns:
    --------
    float : ECE score (lower is better, 0 is perfect)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_proba[mask])
            bin_weight = np.sum(mask) / len(y_true)
            ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
    
    return ece


def calculate_reliability_curve(y_true, y_proba, n_bins=10, strategy='uniform'):
    """
    Calculate reliability (calibration) curve data.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities
    n_bins : int
        Number of bins
    strategy : str
        'uniform' for equal-width bins, 'quantile' for equal-frequency bins
    
    Returns:
    --------
    dict with:
        - 'mean_predicted': mean predicted probability per bin
        - 'fraction_positive': actual fraction of positives per bin
        - 'counts': number of samples per bin
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    if strategy == 'uniform':
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        bin_edges = np.percentile(y_proba, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)  # Remove duplicates
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Use digitize with right=False to handle edge cases properly
    # Values equal to bin_edges[i] go into bin i, except for the last edge
    bin_indices = np.digitize(y_proba, bin_edges, right=False) - 1
    # Handle edge case: values == 1.0 should go into last bin
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
    
    mean_predicted = []
    fraction_positive = []
    counts = []
    
    for i in range(len(bin_edges) - 1):
        mask = bin_indices == i
        n_samples = np.sum(mask)
        if n_samples > 0:
            mean_predicted.append(np.mean(y_proba[mask]))
            fraction_positive.append(np.mean(y_true[mask]))
            counts.append(n_samples)
        else:
            # Skip empty bins - don't add them to the arrays
            # This prevents plotting artifacts with zero counts
            pass
    
    return {
        'mean_predicted': np.array(mean_predicted),
        'fraction_positive': np.array(fraction_positive),
        'counts': np.array(counts),
        'bin_edges': bin_edges
    }


def plot_reliability_curve(y_true, y_proba, n_bins=10, ax=None, label='Model', strategy='quantile'):
    """
    Plot reliability (calibration) curve.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities
    n_bins : int
        Number of bins
    ax : matplotlib axis
        Axis to plot on (creates new if None)
    label : str
        Label for the curve
    strategy : str
        'uniform' for equal-width bins, 'quantile' for equal-frequency bins
    
    Returns:
    --------
    matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    curve_data = calculate_reliability_curve(y_true, y_proba, n_bins=n_bins, strategy=strategy)
    
    # Plot calibration curve
    ax.plot(curve_data['mean_predicted'], curve_data['fraction_positive'], 
            marker='o', label=label, linewidth=2)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Reliability Curve', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return ax


def brier_skill_score(y_true, y_proba):
    """
    Calculate Brier Skill Score (BSS).
    BSS = 1 - (Brier Score / Brier Score of reference model)
    Reference model predicts base rate for all samples.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities
    
    Returns:
    --------
    float : BSS (can be negative if worse than baseline)
    """
    y_true = np.asarray(y_true)
    brier = brier_score_loss(y_true, y_proba)
    base_rate = np.mean(y_true)
    brier_baseline = brier_score_loss(y_true, np.full_like(y_proba, base_rate))
    
    if brier_baseline == 0:
        return np.nan
    
    return 1 - (brier / brier_baseline)


def decompose_brier_score(y_true, y_proba, n_bins=10):
    """
    Decompose Brier score into reliability, resolution, and uncertainty.
    
    Brier = Reliability - Resolution + Uncertainty
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for decomposition
    
    Returns:
    --------
    dict with 'brier', 'reliability', 'resolution', 'uncertainty'
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    brier = brier_score_loss(y_true, y_proba)
    base_rate = np.mean(y_true)
    
    # Bin the predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    reliability = 0.0
    resolution = 0.0
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            n_i = np.sum(mask)
            o_i = np.mean(y_true[mask])  # observed frequency
            f_i = np.mean(y_proba[mask])  # forecast probability
            
            reliability += (n_i / len(y_true)) * (f_i - o_i) ** 2
            resolution += (n_i / len(y_true)) * (o_i - base_rate) ** 2
    
    uncertainty = base_rate * (1 - base_rate)
    
    return {
        'brier': brier,
        'reliability': reliability,
        'resolution': resolution,
        'uncertainty': uncertainty
    }


def print_metrics_report(metrics, title="Model Evaluation Metrics"):
    """
    Print formatted metrics report.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metric names and values
    title : str
        Report title
    """
    print("=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    # Group metrics
    prob_metrics = ['roc_auc', 'pr_auc', 'brier_score', 'log_loss', 'ks_statistic']
    class_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    calib_metrics = ['ece', 'mean_predicted_probability', 'mean_actual_rate']
    
    print("\nProbabilistic Metrics:")
    print("-" * 60)
    for key in prob_metrics:
        if key in metrics:
            print(f"  {key:30s}: {metrics[key]:.4f}")
    
    print("\nClassification Metrics:")
    print("-" * 60)
    for key in class_metrics:
        if key in metrics:
            print(f"  {key:30s}: {metrics[key]:.4f}")
    
    print("\nCalibration Metrics:")
    print("-" * 60)
    for key in calib_metrics:
        if key in metrics:
            print(f"  {key:30s}: {metrics[key]:.4f}")
    
    if 'true_positives' in metrics:
        print("\nConfusion Matrix:")
        print("-" * 60)
        print(f"  True Positives (TP):  {metrics['true_positives']}")
        print(f"  True Negatives (TN):  {metrics['true_negatives']}")
        print(f"  False Positives (FP): {metrics['false_positives']}")
        print(f"  False Negatives (FN): {metrics['false_negatives']}")
    
    print("=" * 60)