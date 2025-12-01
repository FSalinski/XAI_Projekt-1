'''''
Further feature selection using RFE.
'''
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from constants import MAX_FEATURES, TRAIN_PATH, TEST_PATH, TARGET_COLUMN, RANDOM_STATE, TRAIN_TRIMMED_PATH, TEST_TRIMMED_PATH
from data_processing import data_processing_pipeline_feature_selection
import pandas as pd
from utils import load_train_test_data
import logging

# load train and test data
X_train, X_test, y_train, y_test = load_train_test_data()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting feature selection using RFE")
    # Random Forest RFE
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rfe_rf = RFE(estimator=rf, n_features_to_select=MAX_FEATURES, step=5, verbose=2)
    pipeline = Pipeline(steps=[
        ('data_processing', data_processing_pipeline_feature_selection(X_train)),
        ('rfe', rfe_rf)
    ])
    pipeline.fit(X_train, y_train)
    
    # Get feature names after preprocessing
    preprocessed_feature_names = pipeline.named_steps['data_processing'].get_feature_names_out()
    
    # Get selected features mask from RFE
    selected_features_mask = pipeline.named_steps['rfe'].support_
    selected_feature_names = preprocessed_feature_names[selected_features_mask]
    
    # transform train and test sets through the entire pipeline
    X_train_rf = pipeline.transform(X_train)
    X_test_rf = pipeline.transform(X_test)
    
    # convert back to DataFrame with proper feature names
    X_train_rf_df = pd.DataFrame(X_train_rf, columns=selected_feature_names)
    X_test_rf_df = pd.DataFrame(X_test_rf, columns=selected_feature_names)


    # save to csv
    X_train_rf_df[TARGET_COLUMN] = y_train.values
    X_test_rf_df[TARGET_COLUMN] = y_test.values
    X_train_rf_df.to_csv(TRAIN_TRIMMED_PATH, index=False)
    X_test_rf_df.to_csv(TEST_TRIMMED_PATH, index=False)

if __name__ == "__main__":
    main()
