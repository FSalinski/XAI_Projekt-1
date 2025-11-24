import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from utils import (
    save_model,
    load_model,
    load_train_test_data,
    calibration_metrics_dict,
    expected_calibration_error,
    reliability_diagram,
)
from constants import (
    RANDOM_STATE, 
    TRAIN_PATH, 
    TEST_PATH, 
    TUNED_LR_MODEL_PATH, 
    TUNED_RF_MODEL_PATH,
    CALIBRATED_LR_MODEL_PATH, 
    CALIBRATED_RF_MODEL_PATH,
    PLOTS_PATH,
)


def calibrate_logistic_regression(X_val, X_test, y_val, y_test):
    """Calibrate Logistic Regression model."""
    logging.info("=" * 50)
    logging.info("Calibrating Logistic Regression model")
    
    # Load tuned Logistic Regression model
    model = load_model(TUNED_LR_MODEL_PATH)
    logging.info("Loaded tuned Logistic Regression model")
    
    # Get predicted probabilities
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    
    val_metrics = calibration_metrics_dict(y_val, p_val)
    test_metrics = calibration_metrics_dict(y_test, p_test)
    
    logging.info(f"VAL metrics: {val_metrics}")
    logging.info(f"TEST metrics: {test_metrics}")
    
    # Calculate and log ECE for base model
    ece_base = expected_calibration_error(y_test.values, p_test)
    logging.info(f"Base model ECE: {ece_base:.4f}")
    
    # Plot base model reliability diagram
    fig, ax = plt.subplots(figsize=(7, 6))
    reliability_diagram(ax, y_test, p_test, label="base")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_PATH, 'lr_base_reliability.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved base model reliability diagram to {plot_path}")
    
    # Calibrate with sigmoid and isotonic methods
    logging.info("Training sigmoid calibration...")
    sig = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
    sig.fit(X_val, y_val)
    
    logging.info("Training isotonic calibration...")
    iso = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
    iso.fit(X_val, y_val)
    
    # Get calibrated predictions
    p_test_sig = sig.predict_proba(X_test)[:, 1]
    p_test_iso = iso.predict_proba(X_test)[:, 1]
    
    sig_metrics = calibration_metrics_dict(y_test, p_test_sig)
    iso_metrics = calibration_metrics_dict(y_test, p_test_iso)
    
    logging.info(f"Sigmoid calibration metrics: {sig_metrics}")
    logging.info(f"Isotonic calibration metrics: {iso_metrics}")
    
    ece_sig = expected_calibration_error(y_test.values, p_test_sig)
    ece_iso = expected_calibration_error(y_test.values, p_test_iso)
    
    logging.info(f"ECE sigmoid: {ece_sig:.4f}")
    logging.info(f"ECE isotonic: {ece_iso:.4f}")
    
    # Plot comparison of all calibration methods
    fig, ax = plt.subplots(figsize=(8, 6))
    reliability_diagram(ax, y_test, p_test, label="base")
    reliability_diagram(ax, y_test, p_test_sig, label="sigmoid")
    reliability_diagram(ax, y_test, p_test_iso, label="isotonic")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_PATH, 'lr_calibration_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved calibration comparison to {plot_path}")
    
    # Save sigmoid calibrated model
    logging.info("Saving sigmoid calibrated model (chosen for better AUC preservation)")
    save_model(sig, CALIBRATED_LR_MODEL_PATH)
    logging.info(f"Calibrated Logistic Regression model saved to {CALIBRATED_LR_MODEL_PATH}")


def calibrate_random_forest(X_val, X_test, y_val, y_test):
    """Calibrate Random Forest model."""
    logging.info("=" * 50)
    logging.info("Calibrating Random Forest model")
    
    # Load tuned Random Forest model
    model = load_model(TUNED_RF_MODEL_PATH)
    logging.info("Loaded tuned Random Forest model")
    
    # Get predicted probabilities
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    
    val_metrics = calibration_metrics_dict(y_val, p_val)
    test_metrics = calibration_metrics_dict(y_test, p_test)
    
    logging.info(f"VAL metrics: {val_metrics}")
    logging.info(f"TEST metrics: {test_metrics}")
    
    # Calculate and log ECE for base model
    ece_base = expected_calibration_error(y_test.values, p_test)
    logging.info(f"Base model ECE: {ece_base:.4f}")
    
    # Plot base model reliability diagram
    fig, ax = plt.subplots(figsize=(7, 6))
    reliability_diagram(ax, y_test, p_test, label="base")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_PATH, 'rf_base_reliability.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved base model reliability diagram to {plot_path}")
    
    # Calibrate with sigmoid and isotonic methods
    logging.info("Training sigmoid calibration...")
    sig = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
    sig.fit(X_val, y_val)
    
    logging.info("Training isotonic calibration...")
    iso = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
    iso.fit(X_val, y_val)
    
    # Get calibrated predictions
    p_test_sig = sig.predict_proba(X_test)[:, 1]
    p_test_iso = iso.predict_proba(X_test)[:, 1]
    
    sig_metrics = calibration_metrics_dict(y_test, p_test_sig)
    iso_metrics = calibration_metrics_dict(y_test, p_test_iso)
    
    logging.info(f"Sigmoid calibration metrics: {sig_metrics}")
    logging.info(f"Isotonic calibration metrics: {iso_metrics}")
    
    ece_sig = expected_calibration_error(y_test.values, p_test_sig)
    ece_iso = expected_calibration_error(y_test.values, p_test_iso)
    
    logging.info(f"ECE sigmoid: {ece_sig:.4f}")
    logging.info(f"ECE isotonic: {ece_iso:.4f}")
    
    # Plot comparison of all calibration methods
    fig, ax = plt.subplots(figsize=(8, 6))
    reliability_diagram(ax, y_test, p_test, label="base")
    reliability_diagram(ax, y_test, p_test_sig, label="sigmoid")
    reliability_diagram(ax, y_test, p_test_iso, label="isotonic")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_PATH, 'rf_calibration_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved calibration comparison to {plot_path}")
    
    # Save sigmoid calibrated model
    logging.info("Saving sigmoid calibrated model (chosen for better AUC and logloss)")
    save_model(sig, CALIBRATED_RF_MODEL_PATH)
    logging.info(f"Calibrated Random Forest model saved to {CALIBRATED_RF_MODEL_PATH}")


def main():
    # Set up plots directory
    os.makedirs(PLOTS_PATH, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("=" * 50)
    logging.info("Starting model calibration")
    
    # Load data
    X_train, X_test, y_train, y_test = load_train_test_data(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
    )
    logging.info(f"Loaded training data: {X_train.shape}, test data: {X_test.shape}")
    
    # Split test set into validation and test for calibration evaluation
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=RANDOM_STATE, stratify=y_test
    )
    logging.info(f"Split test set - validation: {X_val.shape}, test: {X_test.shape}")
    
    # Calibrate Logistic Regression
    calibrate_logistic_regression(X_val, X_test, y_val, y_test)
    
    # Calibrate Random Forest
    calibrate_random_forest(X_val, X_test, y_val, y_test)
    
    logging.info("=" * 50)
    logging.info("Model calibration completed successfully")


if __name__ == '__main__':
    main()
