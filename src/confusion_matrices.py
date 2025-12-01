from utils import load_model, load_train_test_data
from constants import CALIBRATED_LR_MODEL_PATH, CALIBRATED_RF_MODEL_PATH, TUNED_LR_MODEL_PATH, TUNED_RF_MODEL_PATH, PLOTS_PATH, TEST_TRIMMED_PATH
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import confusion_matrix

def main():
    """Generate and save confusion matrices for tuned models."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("=" * 50)
    logging.info("Generating confusion matrices for tuned models")

    # Load tuned models
    lr_model = load_model(TUNED_LR_MODEL_PATH)
    rf_model = load_model(TUNED_RF_MODEL_PATH)

    # Load test data
    _, X_test, _, y_test = load_train_test_data(test_path=TEST_TRIMMED_PATH)
    # Predict
    y_pred_lr = lr_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    # Confusion matrices
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[0])
    axes[0].set_title('Logistic Regression Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Random Forest Confusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_PATH}/confusion_matrices.png")
    plt.close()

if __name__ == "__main__":
    main()