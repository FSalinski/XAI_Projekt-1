from utils import load_model, load_train_test_data
from constants import CALIBRATED_LR_MODEL_PATH, CALIBRATED_RF_MODEL_PATH, TUNED_LR_MODEL_PATH, TUNED_RF_MODEL_PATH, PLOTS_PATH, TEST_TRIMMED_PATH
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

def main(threshold_lr=0.1250, threshold_rf=0.0650):
    """Generate and save confusion matrices for calibrated models.
    
    Args:
        threshold_lr: Optimal threshold for logistic regression (default 0.5)
        threshold_rf: Optimal threshold for random forest (default 0.5)
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("=" * 50)
    logging.info("Generating confusion matrices for calibrated models")
    logging.info(f"Using threshold for LR: {threshold_lr:.4f}")
    logging.info(f"Using threshold for RF: {threshold_rf:.4f}")

    # Load calibrated models
    lr_model = load_model(CALIBRATED_LR_MODEL_PATH)
    rf_model = load_model(CALIBRATED_RF_MODEL_PATH)

    # Load test data
    _, X_test, _, y_test = load_train_test_data(test_path=TEST_TRIMMED_PATH)
    
    # Predict with custom thresholds
    y_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    y_pred_lr = (y_proba_lr >= threshold_lr).astype(int)
    
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_proba_rf >= threshold_rf).astype(int)
    
    # Confusion matrices
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[0])
    axes[0].set_title(f'Logistic Regression (threshold={threshold_lr:.4f})')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title(f'Random Forest (threshold={threshold_rf:.4f})')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}/confusion_matrices.png")
    plt.close()
    
    logging.info(f"Confusion matrices saved to {PLOTS_PATH}/confusion_matrices.png")
    logging.info("=" * 50)

if __name__ == "__main__":
    main()