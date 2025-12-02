"""
Evaluate tuned models on both train and test sets.
"""
import logging
from utils import load_model, evaluate_model, load_train_test_data
from constants import TUNED_LR_MODEL_PATH, TUNED_RF_MODEL_PATH, TRAIN_TRIMMED_PATH, TEST_TRIMMED_PATH
import warnings
warnings.filterwarnings("ignore")

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
    lr_train_rec, lr_train_auc = evaluate_model(lr_model, X_train, y_train)
    logging.info(f"Train set - Recall: {lr_train_rec:.4f}, ROC AUC: {lr_train_auc:.4f}")
    
    lr_test_rec, lr_test_auc = evaluate_model(lr_model, X_test, y_test)
    logging.info(f"Test set  - Recall: {lr_test_rec:.4f}, ROC AUC: {lr_test_auc:.4f}")
    
    # Evaluate Random Forest
    logging.info("\n--- Random Forest ---")
    rf_train_rec, rf_train_auc = evaluate_model(rf_model, X_train, y_train)
    logging.info(f"Train set - Recall: {rf_train_rec:.4f}, ROC AUC: {rf_train_auc:.4f}")
    
    rf_test_rec, rf_test_auc = evaluate_model(rf_model, X_test, y_test)
    logging.info(f"Test set  - Recall: {rf_test_rec:.4f}, ROC AUC: {rf_test_auc:.4f}")
    
    logging.info("=" * 50)

if __name__ == "__main__":
    main()
