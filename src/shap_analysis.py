import pandas as pd
import numpy as np
import seaborn as sns
import shap
import matplotlib.pyplot as plt
import logging
import os
import warnings
import sys
from sklearn.inspection import permutation_importance

# Add project root to path to ensure imports work if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_model, load_train_test_data
from src.constants import (
    TRAIN_TRIMMED_PATH,
    TEST_TRIMMED_PATH,
    TUNED_LR_MODEL_PATH, 
    TUNED_RF_MODEL_PATH,
    PROJECT_ROOT,
    RANDOM_STATE
)

warnings.filterwarnings("ignore")

PLOTS_PATH = os.path.join(PROJECT_ROOT, 'plots', 'shap')
os.makedirs(PLOTS_PATH, exist_ok=True)

def analyze_shap_lr(X_train, X_test, y_test):
    logging.info("=" * 50)
    logging.info("Starting SHAP analysis for Logistic Regression")
    
    # Load model
    model = load_model(TUNED_LR_MODEL_PATH)
    logging.info("Loaded tuned Logistic Regression model")
    
    # Extract components
    pipeline_step = model.named_steps['pipeline']
    classifier = model.named_steps['classifier']
    
    # Transform data
    logging.info("Transforming data for SHAP analysis...")
    # The pipeline is now configured to return pandas DataFrames with feature names
    X_train_transformed = pipeline_step.transform(X_train)
    X_test_transformed = pipeline_step.transform(X_test)
    
    # 1. Model's native feature importance (coefficients)
    logging.info("Plotting model coefficients (native feature importance)...")
    coefs = pd.Series(classifier.coef_[0], index=X_train_transformed.columns)
    top_15_coefs = coefs.abs().nlargest(15).index
    coefs_top15 = coefs[top_15_coefs].sort_values()
    
    plt.figure(figsize=(10, 8))
    coefs_top15.plot(kind='barh')
    plt.xlabel('Coefficient Value')
    plt.title('Top 15 Features by Absolute Coefficient - Logistic Regression')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'lr_coefficients.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Top 10 features by absolute coefficient:")
    for idx, (feat, val) in enumerate(coefs.abs().nlargest(10).items(), 1):
        logging.info(f"  {idx}. {feat}: {val:.6f} (original: {coefs[feat]:.6f})")
    
    # Create Explainer
    logging.info("Creating LinearExplainer...")
    # LinearExplainer works well with DataFrames if feature names are present
    explainer = shap.LinearExplainer(classifier, X_train_transformed)
    
    # Calculate SHAP values for test set
    logging.info("Calculating SHAP values...")
    shap_values = explainer(X_test_transformed)
    
    # Get top 15 features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-15:][::-1]
    
    logging.info("Top 15 features by mean absolute SHAP value:")
    for idx, feat_idx in enumerate(top_indices, 1):
        feat_name = shap_values.feature_names[feat_idx]
        logging.info(f"  {idx}. {feat_name}: {mean_abs_shap[feat_idx]:.6f}")
    
    # Create filtered SHAP values for top features only
    shap_values_top = shap.Explanation(
        values=shap_values.values[:, top_indices],
        base_values=shap_values.base_values,
        data=shap_values.data[:, top_indices] if shap_values.data is not None else None,
        feature_names=[shap_values.feature_names[i] for i in top_indices]
    )
    
    # 1. Beeswarm plot (top 15 features only, no aggregation)
    logging.info("Generating Beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values_top, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'lr_shap_beeswarm.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar plot (Global importance, top 15 features only)
    logging.info("Generating Bar plot...")
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values_top, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'lr_shap_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Waterfall plots for multiple observations
    logging.info("Generating Waterfall plots for selected observations...")
    
    # Select 3 interesting observations: high probability default, low probability default, medium
    y_proba = classifier.predict_proba(X_test_transformed)[:, 1]
    
    # Find indices of interesting cases among actual defaults (y_test == 1)
    default_indices = np.where(y_test == 1)[0]
    if len(default_indices) >= 3:
        # Among defaults, find high, medium, and low predicted probabilities
        default_probas = y_proba[default_indices]
        high_idx = default_indices[np.argmax(default_probas)]
        low_idx = default_indices[np.argmin(default_probas)]
        mid_idx = default_indices[np.argsort(default_probas)[len(default_probas)//2]]
        
        obs_indices = [high_idx, mid_idx, low_idx]
        obs_labels = ['high_prob_default', 'medium_prob_default', 'low_prob_default']
    else:
        # Fallback: just take first 3 observations
        obs_indices = [0, 1, 2]
        obs_labels = ['obs0', 'obs1', 'obs2']
    
    for idx, (obs_idx, label) in enumerate(zip(obs_indices, obs_labels)):
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_values[obs_idx], max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_PATH, f'lr_shap_waterfall_{label}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"  Generated waterfall plot for observation {obs_idx} (label: {label}, predicted prob: {y_proba[obs_idx]:.4f}, actual: {y_test.iloc[obs_idx]})")
    
    # 4. Permutation Importance (only for top 20 original features by SHAP)
    logging.info("Calculating Permutation Importance for top 20 original features...")
    
    # Calculate SHAP importance per original feature by summing over transformed features
    # Group transformed features by their original feature names
    original_feature_importance = {}
    for i, feat_name in enumerate(shap_values.feature_names):
        # Extract base feature name (before one-hot encoding or indicator)
        # Handle both one-hot encoding and missing indicators
        if 'missingindicator_' in feat_name.lower():
            # Skip missing indicator columns for permutation importance
            continue
        elif '_' in feat_name:
            # Try to extract base feature name
            base_feat = feat_name.rsplit('_', 1)[0]
        else:
            base_feat = feat_name
        
        # Only consider features that exist in X_test
        if base_feat in X_test.columns:
            if base_feat not in original_feature_importance:
                original_feature_importance[base_feat] = 0
            original_feature_importance[base_feat] += mean_abs_shap[i]
    
    # Get top 20 original features
    original_features_sorted = sorted(original_feature_importance.items(), key=lambda x: x[1], reverse=True)
    original_features_to_permute = [feat for feat, _ in original_features_sorted[:20]]
    
    logging.info(f"Selected {len(original_features_to_permute)} original features for permutation importance")
    
    model_full = load_model(TUNED_LR_MODEL_PATH)
    
    # Compute permutation importance manually
    from sklearn.metrics import roc_auc_score
    
    # Baseline score
    y_pred_proba = model_full.predict_proba(X_test)[:, 1]
    baseline_score = roc_auc_score(y_test, y_pred_proba)
    
    # Compute importance for each feature
    importances = {}
    for feat in original_features_to_permute:
        scores = []
        for _ in range(10):
            X_permuted = X_test.copy()
            X_permuted[feat] = np.random.permutation(X_permuted[feat].values)
            y_pred_permuted = model_full.predict_proba(X_permuted)[:, 1]
            score = roc_auc_score(y_test, y_pred_permuted)
            scores.append(baseline_score - score)
        importances[feat] = np.mean(scores)
    
    pi_series = pd.Series(importances).sort_values(ascending=False)
    
    logging.info("Top 10 features by Permutation Importance:")
    for idx, (feat, val) in enumerate(pi_series.head(10).items(), 1):
        logging.info(f"  {idx}. {feat}: {val:.6f}")
    
    # Plot Permutation Importance for top 10
    logging.info("Generating Permutation Importance plot...")
    plt.figure(figsize=(10, 8))
    pi_series.head(10).sort_values().plot(kind='barh')
    plt.xlabel('Permutation Importance (ROC AUC Drop)')
    plt.title('Top 10 Features by Permutation Importance - Logistic Regression')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'lr_permutation_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Note: SHAP interaction values are not available for LinearExplainer
    # Linear models assume feature independence (no interactions by design)
    logging.info("SHAP interactions skipped for Linear model (interactions not supported)")

    logging.info("SHAP analysis for Logistic Regression completed.")

def analyze_shap_rf(X_train, X_test, y_test):
    logging.info("=" * 50)
    logging.info("Starting SHAP analysis for Random Forest")
    
    # Load model
    model = load_model(TUNED_RF_MODEL_PATH)
    logging.info("Loaded tuned Random Forest model")
    
    # Extract components
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    
    # Transform data
    logging.info("Transforming data for SHAP analysis...")
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # 1. Model's native feature importance
    logging.info("Plotting model's native feature importance...")
    feature_importances = pd.Series(
        classifier.feature_importances_,
        index=X_train_transformed.columns
    )
    top_15_importance = feature_importances.nlargest(15).sort_values()
    
    plt.figure(figsize=(10, 8))
    top_15_importance.plot(kind='barh')
    plt.xlabel('Feature Importance (Gini Importance)')
    plt.title('Top 15 Features by Native Importance - Random Forest')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'rf_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Top 10 features by native feature importance:")
    for idx, (feat, val) in enumerate(feature_importances.nlargest(10).items(), 1):
        logging.info(f"  {idx}. {feat}: {val:.6f}")
    
    # Create Explainer
    logging.info("Creating TreeExplainer...")
    
    # Sample background data if training set is large
    if X_train_transformed.shape[0] > 100:
        background_data = shap.utils.sample(X_train_transformed, 100, random_state=RANDOM_STATE)
    else:
        background_data = X_train_transformed

    explainer = shap.TreeExplainer(classifier, data=background_data, feature_perturbation="interventional")
    
    # Calculate SHAP values for test set
    logging.info("Calculating SHAP values...")
    
    shap_values_all = explainer(X_test_transformed)
    
    # Check shape to handle binary classification correctly
    if len(shap_values_all.values.shape) == 3 and shap_values_all.values.shape[2] == 2:
        # Create a new Explanation object for just the positive class
        shap_values = shap.Explanation(
            values=shap_values_all.values[:, :, 1],
            base_values=shap_values_all.base_values[:, 1],
            data=X_test_transformed,
            feature_names=X_test_transformed.columns
        )
    else:
        shap_values = shap_values_all

    # Get top 15 features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-15:][::-1]
    
    logging.info("Top 15 features by mean absolute SHAP value:")
    for idx, feat_idx in enumerate(top_indices, 1):
        feat_name = shap_values.feature_names[feat_idx]
        logging.info(f"  {idx}. {feat_name}: {mean_abs_shap[feat_idx]:.6f}")
    
    # Create filtered SHAP values for top features only
    # Handle DataFrame vs array indexing properly
    if isinstance(shap_values.data, pd.DataFrame):
        data_subset = shap_values.data.iloc[:, top_indices]
    elif shap_values.data is not None:
        data_subset = shap_values.data[:, top_indices]
    else:
        data_subset = None
    
    shap_values_top = shap.Explanation(
        values=shap_values.values[:, top_indices],
        base_values=shap_values.base_values,
        data=data_subset,
        feature_names=[shap_values.feature_names[i] for i in top_indices]
    )
    
    # 1. Beeswarm plot (top 15 features only, no aggregation)
    logging.info("Generating Beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values_top, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'rf_shap_beeswarm.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar plot (Global importance, top 15 features only)
    logging.info("Generating Bar plot...")
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values_top, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'rf_shap_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Waterfall plots for multiple observations
    logging.info("Generating Waterfall plots for selected observations...")
    
    # Select 3 interesting observations: high probability default, low probability default, medium
    y_proba = classifier.predict_proba(X_test_transformed)[:, 1]
    
    # Find indices of interesting cases among actual defaults (y_test == 1)
    default_indices = np.where(y_test == 1)[0]
    if len(default_indices) >= 3:
        # Among defaults, find high, medium, and low predicted probabilities
        default_probas = y_proba[default_indices]
        high_idx = default_indices[np.argmax(default_probas)]
        low_idx = default_indices[np.argmin(default_probas)]
        mid_idx = default_indices[np.argsort(default_probas)[len(default_probas)//2]]
        
        obs_indices = [high_idx, mid_idx, low_idx]
        obs_labels = ['high_prob_default', 'medium_prob_default', 'low_prob_default']
    else:
        # Fallback: just take first 3 observations
        obs_indices = [0, 1, 2]
        obs_labels = ['obs0', 'obs1', 'obs2']
    
    for idx, (obs_idx, label) in enumerate(zip(obs_indices, obs_labels)):
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_values[obs_idx], max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_PATH, f'rf_shap_waterfall_{label}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"  Generated waterfall plot for observation {obs_idx} (label: {label}, predicted prob: {y_proba[obs_idx]:.4f}, actual: {y_test.iloc[obs_idx]})")
    
    # 4. Permutation Importance (only for top 20 original features by SHAP)
    logging.info("Calculating Permutation Importance for top 20 original features...")
    
    # Calculate SHAP importance per original feature by summing over transformed features
    original_feature_importance = {}
    for i, feat_name in enumerate(shap_values.feature_names):
        # Extract base feature name (before one-hot encoding or indicator)
        # Handle both one-hot encoding and missing indicators
        if 'missingindicator_' in feat_name.lower():
            # Skip missing indicator columns for permutation importance
            continue
        elif '_' in feat_name:
            # Try to extract base feature name
            base_feat = feat_name.rsplit('_', 1)[0]
        else:
            base_feat = feat_name
        
        # Only consider features that exist in X_test
        if base_feat in X_test.columns:
            if base_feat not in original_feature_importance:
                original_feature_importance[base_feat] = 0
            original_feature_importance[base_feat] += mean_abs_shap[i]
    
    # Get top 20 original features
    original_features_sorted = sorted(original_feature_importance.items(), key=lambda x: x[1], reverse=True)
    original_features_to_permute = [feat for feat, _ in original_features_sorted[:20]]
    
    logging.info(f"Selected {len(original_features_to_permute)} original features for permutation importance")
    
    model_full = load_model(TUNED_RF_MODEL_PATH)
    
    # Compute permutation importance manually
    from sklearn.metrics import roc_auc_score
    
    # Baseline score
    y_pred_proba = model_full.predict_proba(X_test)[:, 1]
    baseline_score = roc_auc_score(y_test, y_pred_proba)
    
    # Compute importance for each feature
    importances = {}
    for feat in original_features_to_permute:
        scores = []
        for _ in range(10):
            X_permuted = X_test.copy()
            X_permuted[feat] = np.random.permutation(X_permuted[feat].values)
            y_pred_permuted = model_full.predict_proba(X_permuted)[:, 1]
            score = roc_auc_score(y_test, y_pred_permuted)
            scores.append(baseline_score - score)
        importances[feat] = np.mean(scores)
    
    pi_series = pd.Series(importances).sort_values(ascending=False)
    
    logging.info("Top 10 features by Permutation Importance:")
    for idx, (feat, val) in enumerate(pi_series.head(10).items(), 1):
        logging.info(f"  {idx}. {feat}: {val:.6f}")
    
    # Plot Permutation Importance for top 10
    logging.info("Generating Permutation Importance plot...")
    plt.figure(figsize=(10, 8))
    pi_series.head(10).sort_values().plot(kind='barh')
    plt.xlabel('Permutation Importance (ROC AUC Drop)')
    plt.title('Top 10 Features by Permutation Importance - Random Forest')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'rf_permutation_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logging.info("SHAP analysis for Random Forest completed.")

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    sns.set_style("whitegrid")
    
    logging.info("=" * 50)
    logging.info("Starting SHAP analysis pipeline")
    
    # Load data
    X_train, X_test, y_train, y_test = load_train_test_data(
        train_path=TRAIN_TRIMMED_PATH,
        test_path=TEST_TRIMMED_PATH,
    )
    logging.info(f"Loaded training data: {X_train.shape}, test data: {X_test.shape}")
    
    # Analyze LR
    analyze_shap_lr(X_train, X_test, y_test)
    
    # Analyze RF
    analyze_shap_rf(X_train, X_test, y_test)
    
    logging.info("=" * 50)
    logging.info("SHAP analysis pipeline completed successfully")

if __name__ == '__main__':
    main()
