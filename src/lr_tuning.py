import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .utils import evaluate_model, save_model, load_train_test_data
from .data_processing import data_processing_pipeline_lr
import logging
from .constants import RANDOM_STATE, MODELS_PATH, TUNED_LR_MODEL_PATH
import os
import optuna

import warnings
warnings.filterwarnings("ignore")

N_TRIALS = 400
N_SPLITS = 5

def main():
    # Set up save directory
    os.makedirs(MODELS_PATH, exist_ok=True)

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("=" * 50)
    logging.info("Starting Logistic Regression hyperparameter tuning with Optuna")

    # Load data
    X_train, X_test, y_train, y_test = load_train_test_data()

    def objective(trial):
        # Suggest hyperparameters
        C = trial.suggest_float('C', 1e-3, 1e3, log=True)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        imputer_strategy = trial.suggest_categorical('imputer_strategy', ['mean', 'median'])
        add_indicator = trial.suggest_categorical('add_indicator', [True, False])
        alpha = trial.suggest_float('alpha', 0.0, 0.15)
        corr_threshold = trial.suggest_float('corr_threshold', 0.5, 1.0)
        
        # Build pipeline with suggested hyperparameters
        pipeline = Pipeline(steps=[
            ('pipeline', data_processing_pipeline_lr(
                X_train, 
                corr_threshold=corr_threshold,
                imputer_strategy=imputer_strategy,
                add_indicator=add_indicator,
                alpha=alpha
            )),
            ('classifier', LogisticRegression(
                penalty='l2', 
                C=C,
                class_weight=class_weight,
                random_state=RANDOM_STATE, 
                solver='liblinear',
                max_iter=1000
            ))
        ])
        
        # Evaluate with cross-validation
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()
    
    # Create Optuna study
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, n_jobs=-1)
    
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best ROC AUC: {study.best_value}")
    logging.info(f"Best hyperparameters: {study.best_params}")
    
    # Train final model with best hyperparameters
    best_params = study.best_params
    final_pipeline = Pipeline(steps=[
        ('pipeline', data_processing_pipeline_lr(
            X_train, 
            corr_threshold=best_params['corr_threshold'],
            imputer_strategy=best_params['imputer_strategy'],
            add_indicator=best_params['add_indicator'],
            alpha=best_params['alpha']
        )),
        ('classifier', LogisticRegression(
            penalty='l2', 
            C=best_params['C'],
            class_weight=best_params['class_weight'],
            random_state=RANDOM_STATE, 
            solver='liblinear',
            max_iter=1000
        ))
    ])
    
    final_pipeline.fit(X_train, y_train)
    rec, roc_auc = evaluate_model(final_pipeline, X_test, y_test)
    logging.info("Results on test set:")
    logging.info(f"Tuned Logistic Regression - Recall: {rec}, ROC AUC: {roc_auc}")

    save_model(final_pipeline, TUNED_LR_MODEL_PATH)
    logging.info(f"Logistic Regression hyperparameter tuning completed. Saved tuned model in '{TUNED_LR_MODEL_PATH}'")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()