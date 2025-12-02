"""
Evaluate tuned models on both train and test sets.
"""
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from utils import load_model, load_train_test_data
from constants import TUNED_LR_MODEL_PATH, TUNED_RF_MODEL_PATH, TRAIN_TRIMMED_PATH, TEST_TRIMMED_PATH
import warnings
warnings.filterwarnings("ignore")

def calculate_ks_statistic(y_true, y_proba):
    """Calculate Kolmogorov-Smirnov statistic."""
    # Sort by predicted probability
    df = np.column_stack([y_true, y_proba])
    df = df[df[:, 1].argsort()[::-1]]
    
    # Calculate cumulative distributions
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    cum_pos = np.cumsum(df[:, 0] == 1) / n_pos
    cum_neg = np.cumsum(df[:, 0] == 0) / n_neg
    
    # KS is the maximum difference between cumulative distributions
    ks = np.max(np.abs(cum_pos - cum_neg))
    return ks

def evaluate_model_comprehensive(model, X, y, set_name=""):
    """Evaluate model with comprehensive metrics."""
    y_proba = model.predict_proba(X)[:, 1]
    
    roc_auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    ks = calculate_ks_statistic(y, y_proba)
    logloss = log_loss(y, y_proba)
    
    logging.info(f"{set_name} set:")
    logging.info(f"  ROC AUC:  {roc_auc:.4f}")
    logging.info(f"  PR AUC:   {pr_auc:.4f}")
    logging.info(f"  KS:       {ks:.4f}")
    logging.info(f"  Log Loss: {logloss:.4f}")
    
    return roc_auc, pr_auc, ks, logloss

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("=" * 50)
    logging.info("Evaluating tuned models")
    
    # Load data
    X_train, X_test, y_train, y_test = load_train_test_data(train_path=TRAIN_TRIMMED_PATH, test_path=TEST_TRIMMED_PATH)
    
    # Load tuned models
    lr_model = load_model(TUNED_LR_MODEL_PATH)
    rf_model = load_model(TUNED_RF_MODEL_PATH)
    
    # Evaluate Logistic Regression
    logging.info("\n--- Logistic Regression ---")
    evaluate_model_comprehensive(lr_model, X_train, y_train, "Train")
    evaluate_model_comprehensive(lr_model, X_test, y_test, "Test")
    
    # Evaluate Random Forest
    logging.info("\n--- Random Forest ---")
    evaluate_model_comprehensive(rf_model, X_train, y_train, "Train")
    evaluate_model_comprehensive(rf_model, X_test, y_test, "Test")
    
    logging.info("=" * 50)

if __name__ == "__main__":
    main()
