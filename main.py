'''
Main module for the project
'''

import logging


SKIP_HYPERPARAMETER_TUNING = True # Whether to skip time-consuming hyperparameter tuning and use pre-trained models saved in the 'models/' directory.

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Main module started")
    logging.info("=" * 50)
    # Step 1: Manual preprocessing
    import manual_preprocessing
    manual_preprocessing.main()
    # Step 2: Feature selection
    import feature_selection
    feature_selection.main()
    # Step 3: Data splitting
    import split_data
    split_data.main()
    # Step 4: Hyperparameter tuning
    if not SKIP_HYPERPARAMETER_TUNING:
        import lr_tuning
        lr_tuning.main()
        import rf_tuning
        rf_tuning.main()
    else:
        logging.info("Skipping hyperparameter tuning as per configuration.")
    # Step 5: Evaluate tuned models
    import evaluate_tuned_models
    evaluate_tuned_models.main()
    # Step 6: Calibrate models
    import calibration
    calibration.main()
    # Step 7: Threshold selection
    import threshold_selection
    threshold_lr, threshold_rf = threshold_selection.main()
    # Step 8: Generate confusion matrices with optimal thresholds
    import confusion_matrices
    confusion_matrices.main(threshold_lr=threshold_lr, threshold_rf=threshold_rf)
    # Step 9: SHAP analysis
    import shap_analysis
    shap_analysis.main()


if __name__ == "__main__":
    main()
