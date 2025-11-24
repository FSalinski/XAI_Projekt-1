'''
Script for training 4 baseline models:
 - Logistic Regression on full feature set
 - Logistic Regression on reduced feature set (after removing pairwise correlated features)
 - Random Forest on full feature set
 - Random Forest on reduced feature set (after removing pairwise correlated features)
'''

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils import evaluate_model, load_train_test_data, save_model
from data_processing import data_processing_pipeline_lr
import logging
from constants import RANDOM_STATE
import os

CORR_THRESHOLD = 0.75

def main():
    # set up save directory
    os.makedirs('./models', exist_ok=True)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("=" * 50)
    logging.info("Starting baseline model training")

    # Load data
    X_train, X_test, y_train, y_test = load_train_test_data()

    # Baseline data processing pipeline
    preprocessor = data_processing_pipeline_lr(X_train, corr_threshold=2)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Create dataset with reduced features (removing pairwise correlated features)
    preprocessor_reduced = data_processing_pipeline_lr(X_train, corr_threshold=CORR_THRESHOLD)
    X_train_reduced = preprocessor_reduced.fit_transform(X_train)
    X_test_reduced = preprocessor_reduced.transform(X_test)

    # Logistic Regression and Random Forest models
    lr = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', solver='liblinear')
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')

    # Logistic Regression on full feature set
    log_reg_full = lr
    log_reg_full.fit(X_train_processed, y_train)
    rec_full_lr, roc_auc_full_lr = evaluate_model(log_reg_full, X_test_processed, y_test)
    logging.info(f"Logistic Regression (full features) - Recall: {rec_full_lr}, ROC AUC: {roc_auc_full_lr}")
    save_model(log_reg_full, './models/logistic_regression_full.pkl')

    # Logistic Regression on reduced feature set
    log_reg_reduced = lr
    log_reg_reduced.fit(X_train_reduced, y_train)
    rec_reduced_lr, roc_auc_reduced_lr = evaluate_model(log_reg_reduced, X_test_reduced, y_test)
    logging.info(f"Logistic Regression (reduced features) - Recall: {rec_reduced_lr}, ROC AUC: {roc_auc_reduced_lr}")
    save_model(log_reg_reduced, './models/logistic_regression_reduced.pkl')

    # Random Forest on full feature set
    rf_full = rf
    rf_full.fit(X_train_processed, y_train)
    rec_full_rf, roc_auc_full_rf = evaluate_model(rf_full, X_test_processed, y_test)
    logging.info(f"Random Forest (full features) - Recall: {rec_full_rf}, ROC AUC: {roc_auc_full_rf}")
    save_model(rf_full, './models/random_forest_full.pkl')

    # Random Forest on reduced feature set
    rf_reduced = rf
    rf_reduced.fit(X_train_reduced, y_train)
    rec_reduced_rf, roc_auc_reduced_rf = evaluate_model(rf_reduced, X_test_reduced, y_test)
    logging.info(f"Random Forest (reduced features) - Recall: {rec_reduced_rf}, ROC AUC: {roc_auc_reduced_rf}")
    save_model(rf_reduced, './models/random_forest_reduced.pkl')
    logging.info("=" * 50)

if __name__ == "__main__":
    main()
