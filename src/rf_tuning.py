import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from utils import evaluate_model, save_model, load_model
from data_processing import data_processing_pipeline_rf
import logging
from constants import RANDOM_STATE, TEST_SIZE
import os
import optuna

import warnings
warnings.filterwarnings("ignore")

N_TRIALS = 400
N_SPLITS = 5

def main():
    # Set up save directory
    os.makedirs('./models', exist_ok=True)
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("=" * 50)
    logging.info("Starting Random Forest hyperparameter tuning with Optuna")

    # Load data
    train = pd.read_csv('data/train.csv')
    X_train = train.drop(columns=['default'])
    y_train = train['default']
    test = pd.read_csv('data/test.csv')
    X_test = test.drop(columns=['default'])
    y_test = test['default']

    def objective(trial):
        # Suggest hyperparameters
        n_estimators = trial.suggest_int('n_estimators', 50, 300, step=5)
        max_depth = trial.suggest_int('max_depth', 3, 12)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
        imputer_strategy = trial.suggest_categorical('imputer_strategy', ['mean', 'median'])
        add_indicator = trial.suggest_categorical('add_indicator', [True, False])
        
        # Build pipeline with suggested hyperparameters
        pipeline = Pipeline(steps=[
            ('preprocessor', data_processing_pipeline_rf(
                X_train,
                corr_threshold=0.75,
                imputer_strategy=imputer_strategy,
                add_indicator=add_indicator
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1
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
    logging.info(f"Best cross-validation mean ROC AUC: {study.best_value}")
    logging.info(f"Best hyperparameters: {study.best_params}")
    
    # Train final model with best hyperparameters
    best_params = study.best_params
    final_pipeline = Pipeline(steps=[
        ('preprocessor', data_processing_pipeline_rf(
            X_train,
            corr_threshold=0.75,
            imputer_strategy=best_params['imputer_strategy'],
            add_indicator=best_params['add_indicator']
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_leaf=best_params['min_samples_leaf'],
            class_weight=best_params['class_weight'],
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    
    final_pipeline.fit(X_train, y_train)
    rec, roc_auc = evaluate_model(final_pipeline, X_test, y_test)
    logging.info("Results on test set:")
    logging.info(f"Tuned Random Forest - Recall: {rec}, ROC AUC: {roc_auc}")

    save_model(final_pipeline, './models/tuned_random_forest.pkl')
    
    logging.info("Random Forest hyperparameter tuning completed. Saved tuned model in './models/tuned_random_forest.pkl'")
    logging.info("=" * 50)

if __name__ == "__main__":
    main()