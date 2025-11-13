'''
Script for training 4 baseline models:
 - Logistic Regression on full feature set
 - Logistic Regression on reduced feature set (after removing correlated features)
 - Random Forest on full feature set
 - Random Forest on reduced feature set (after removing correlated features)
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from utils import remove_all_correlated_features, evaluate_baseline_model, save_model
from data_processing import baseline_data_processing_pipeline
import logging

RANDOM_STATE = 2137

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("=" * 50)
    logging.info("Starting baseline model training")

    # Load data
    df = pd.read_csv('./data/zbiór_5.csv')
    X = df.drop(columns=['default'])
    y = df['default']
    logging.info("Data loaded successfully")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    logging.info("Data split into training and testing sets")

    # Baseline data processing pipeline
    preprocessor = baseline_data_processing_pipeline(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    logging.info("Data processing pipeline applied")

    # Logistic Regression on full feature set
    log_reg_full = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    log_reg_full.fit(X_train_processed, y_train)
    acc_full, roc_auc_full = evaluate_baseline_model(log_reg_full, X_test_processed, y_test)
    logging.info(f"Logistic Regression (full features) - Accuracy: {acc_full}, ROC AUC: {roc_auc_full}")
    # save_model(log_reg_full, './models/logistic_regression_full.pkl')

    # Logistic Regression on reduced feature set
    X_train_reduced = remove_all_correlated_features(pd.DataFrame(X_train_processed), threshold=0.75)
    X_test_reduced = X_test_processed[:, X_train_reduced.columns]
    log_reg_reduced = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    log_reg_reduced.fit(X_train_reduced, y_train)
    acc_reduced, roc_auc_reduced = evaluate_baseline_model(log_reg_reduced, X_test_reduced, y_test)
    logging.info(f"Logistic Regression (reduced features) - Accuracy: {acc_reduced}, ROC AUC: {roc_auc_reduced}")
    # save_model(log_reg_reduced, './models/logistic_regression_reduced.pkl')

    # Random Forest on full feature set
    rf_full = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_full.fit(X_train_processed, y_train)
    acc_rf_full, roc_auc_rf_full = evaluate_baseline_model(rf_full, X_test_processed, y_test)
    logging.info(f"Random Forest (full features) - Accuracy: {acc_rf_full}, ROC AUC: {roc_auc_rf_full}")
    # save_model(rf_full, './models/random_forest_full.pkl')

    # Random Forest on reduced feature set
    rf_reduced = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_reduced.fit(X_train_reduced, y_train)
    acc_rf_reduced, roc_auc_rf_reduced = evaluate_baseline_model(rf_reduced, X_test_reduced, y_test)
    logging.info(f"Random Forest (reduced features) - Accuracy: {acc_rf_reduced}, ROC AUC: {roc_auc_rf_reduced}")
    # save_model(rf_reduced, './models/random_forest_reduced.pkl')
    logging.info("=" * 50)

if __name__ == "__main__":
    main()
