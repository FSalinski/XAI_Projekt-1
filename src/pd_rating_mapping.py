"""
PD to Rating Mapping
Maps probability of default (PD) values to credit rating categories.
"""

import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from constants import TEST_PATH, RANDOM_STATE, CALIBRATED_LR_MODEL_PATH, CALIBRATED_RF_MODEL_PATH, PLOTS_PATH
from utils import load_model


def get_rating_from_pd(pd_value):
    """
    Map PD value to rating category.
    
    Rating scale based on typical credit rating distributions:
    - AAA: 0.00% - 0.03%   (exceptionally strong)
    - AA:  0.03% - 0.10%   (very strong)
    - A:   0.10% - 0.40%   (strong)
    - BBB: 0.40% - 1.50%   (adequate)
    - BB:  1.50% - 6.00%   (speculative)
    - B:   6.00% - 20.00%  (highly speculative)
    - CCC: 20.00% - 50.00% (substantial risk)
    - CC:  50.00% - 80.00% (very high risk)
    - C:   80.00% - 100.00% (near default)
    
    Args:
        pd_value: Probability of default (0-1 scale)
    
    Returns:
        Rating string (AAA, AA, A, BBB, BB, B, CCC, CC, C)
    """
    pd_pct = pd_value * 100
    
    if pd_pct < 0.03:
        return 'AAA'
    elif pd_pct < 0.10:
        return 'AA'
    elif pd_pct < 0.40:
        return 'A'
    elif pd_pct < 1.50:
        return 'BBB'
    elif pd_pct < 6.00:
        return 'BB'
    elif pd_pct < 20.00:
        return 'B'
    elif pd_pct < 50.00:
        return 'CCC'
    elif pd_pct < 80.00:
        return 'CC'
    else:
        return 'C'


def create_rating_distribution_plot(y_pred_lr, y_pred_rf, y_test):
    """Create plot showing rating distribution for both models."""
    
    # Get ratings
    ratings_lr = [get_rating_from_pd(pd) for pd in y_pred_lr]
    ratings_rf = [get_rating_from_pd(pd) for pd in y_pred_rf]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Rating_LR': ratings_lr,
        'Rating_RF': ratings_rf,
        'Actual_Default': y_test
    })
    
    # Rating order for plotting
    rating_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C']
    
    # Count distributions
    lr_counts = df['Rating_LR'].value_counts().reindex(rating_order, fill_value=0)
    rf_counts = df['Rating_RF'].value_counts().reindex(rating_order, fill_value=0)
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Logistic Regression
    axes[0].bar(rating_order, lr_counts.values, color='steelblue', alpha=0.7)
    axes[0].set_title('Rozkład ratingów - Regresja Logistyczna', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Liczba obserwacji')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Random Forest
    axes[1].bar(rating_order, rf_counts.values, color='forestgreen', alpha=0.7)
    axes[1].set_title('Rozkład ratingów - Las Losowy', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Rating')
    axes[1].set_ylabel('Liczba obserwacji')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'rating_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Rating distribution plot saved to {os.path.join(PLOTS_PATH, 'rating_distribution.png')}")
    
    return df, lr_counts, rf_counts


def create_rating_default_rate_plot(df):
    """Create plot showing actual default rate by rating category."""
    
    rating_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Logistic Regression
    lr_default_rates = df.groupby('Rating_LR')['Actual_Default'].agg(['mean', 'count'])
    lr_default_rates = lr_default_rates.reindex(rating_order)
    lr_default_rates = lr_default_rates[lr_default_rates['count'] > 0]
    
    axes[0].bar(lr_default_rates.index, lr_default_rates['mean'] * 100, color='steelblue', alpha=0.7)
    axes[0].set_title('Faktyczna stopa defaultu wg ratingu - LR', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Stopa defaultu (%)')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (idx, row) in enumerate(lr_default_rates.iterrows()):
        axes[0].text(i, row['mean'] * 100 + 0.5, f"n={int(row['count'])}", 
                    ha='center', va='bottom', fontsize=8)
    
    # Random Forest
    rf_default_rates = df.groupby('Rating_RF')['Actual_Default'].agg(['mean', 'count'])
    rf_default_rates = rf_default_rates.reindex(rating_order)
    rf_default_rates = rf_default_rates[rf_default_rates['count'] > 0]
    
    axes[1].bar(rf_default_rates.index, rf_default_rates['mean'] * 100, color='forestgreen', alpha=0.7)
    axes[1].set_title('Faktyczna stopa defaultu wg ratingu - RF', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Rating')
    axes[1].set_ylabel('Stopa defaultu (%)')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (idx, row) in enumerate(rf_default_rates.iterrows()):
        axes[1].text(i, row['mean'] * 100 + 0.5, f"n={int(row['count'])}", 
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, 'rating_default_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Rating default rates plot saved to {os.path.join(PLOTS_PATH, 'rating_default_rates.png')}")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("=" * 50)
    logging.info("Starting PD to Rating Mapping")
    
    # Load test data
    test_data = pd.read_csv(TEST_PATH)
    X_test = test_data.drop('default', axis=1)
    y_test = test_data['default']
    logging.info(f"Loaded test data: {X_test.shape}")
    
    # Load models
    lr_model = load_model(CALIBRATED_LR_MODEL_PATH)
    rf_model = load_model(CALIBRATED_RF_MODEL_PATH)
    logging.info("Models loaded successfully")
    
    # Get predictions
    y_pred_lr = lr_model.predict_proba(X_test)[:, 1]
    y_pred_rf = rf_model.predict_proba(X_test)[:, 1]
    logging.info("Predictions generated")
    
    # Create rating distributions
    df, lr_counts, rf_counts = create_rating_distribution_plot(y_pred_lr, y_pred_rf, y_test)
    
    logging.info("\nRozkład ratingów - Regresja Logistyczna:")
    for rating, count in lr_counts.items():
        if count > 0:
            pct = (count / len(y_test)) * 100
            logging.info(f"  {rating}: {count} ({pct:.1f}%)")
    
    logging.info("\nRozkład ratingów - Las Losowy:")
    for rating, count in rf_counts.items():
        if count > 0:
            pct = (count / len(y_test)) * 100
            logging.info(f"  {rating}: {count} ({pct:.1f}%)")
    
    # Create default rate by rating plot
    create_rating_default_rate_plot(df)
    
    # Log summary statistics
    logging.info("\nStatystyki PD:")
    logging.info(f"  LR - średnia PD: {y_pred_lr.mean():.4f}, mediana: {np.median(y_pred_lr):.4f}")
    logging.info(f"  RF - średnia PD: {y_pred_rf.mean():.4f}, mediana: {np.median(y_pred_rf):.4f}")
    logging.info(f"  Faktyczny default rate: {y_test.mean():.4f}")
    
    logging.info("=" * 50)
    logging.info("PD to Rating Mapping completed successfully")


if __name__ == '__main__':
    main()
