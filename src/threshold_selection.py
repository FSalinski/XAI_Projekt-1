import seaborn as sns
import shap
import matplotlib.pyplot as plt
import logging
import os
import warnings
import sys
from sklearn.inspection import permutation_importance
import numpy as np

# Add project root to path to ensure imports work if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_model, load_train_test_data
from src.constants import (
    CALIBRATED_LR_MODEL_PATH,
    CALIBRATED_RF_MODEL_PATH,
    TEST_TRIMMED_PATH,
    TRAIN_TRIMMED_PATH, 
    PROJECT_ROOT,
    COST_TP,
    COST_FN,
    COST_FP,
    COST_TN,
)

warnings.filterwarnings("ignore")

PLOTS_PATH = os.path.join(PROJECT_ROOT, 'plots', 'shap')
os.makedirs(PLOTS_PATH, exist_ok=True)

# Load data
X_train, X_test, y_train, y_test = load_train_test_data(train_path=TRAIN_TRIMMED_PATH, test_path=TEST_TRIMMED_PATH)

def cost_for_threshold(y_true, p, thr):
    yhat = (p >= thr).astype(int)
    tp = np.sum((yhat==1) & (y_true==1))
    fp = np.sum((yhat==1) & (y_true==0))
    fn = np.sum((yhat==0) & (y_true==1))
    tn = np.sum((yhat==0) & (y_true==0))
    return tp*COST_TP + fp*COST_FP + fn*COST_FN + tn*COST_TN, tp, fp, fn, tn

def sweep_costs(y_true, p, n=101):
    thrs = np.linspace(0,1,n)
    costs, details = [], []
    for t in thrs:
        c, tp, fp, fn, tn = cost_for_threshold(y_true, p, t)
        costs.append(c); details.append((tp,fp,fn,tn))
    return thrs, np.array(costs), details

def main():
    sns.set_style("whitegrid")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("=" * 50)
    logging.info("Finding optimal thresholds for both models")
    
    # Logistic Regression
    logging.info("\n--- Logistic Regression ---")
    lr_model = load_model(CALIBRATED_LR_MODEL_PATH)
    p_lr = lr_model.predict_proba(X_test)[:, 1]
    thrs_lr, costs_lr, details_lr = sweep_costs(y_test, p_lr, n=201)
    best_idx_lr = int(np.argmin(costs_lr))
    best_thr_lr = float(thrs_lr[best_idx_lr])
    best_tp_lr, best_fp_lr, best_fn_lr, best_tn_lr = details_lr[best_idx_lr]

    plt.figure()
    plt.plot(thrs_lr, costs_lr)
    plt.title("Krzywa kosztu vs próg - Regresja Logistyczna")
    plt.xlabel("Próg")
    plt.ylabel("Koszt (niżej lepiej)")
    plt.savefig(f"{PLOTS_PATH}/cost_curve_lr.png")
    plt.close()

    logging.info(f"Najlepszy próg LR: {best_thr_lr:.4f}")
    logging.info(f"TP, FP, FN, TN: {best_tp_lr}, {best_fp_lr}, {best_fn_lr}, {best_tn_lr}")
    accept_rate_lr = (best_tn_lr + best_fn_lr) / len(y_test)
    logging.info(f"Stopa akceptacji przy najlepszym progu: {accept_rate_lr:.4f}")
    
    # Random Forest
    logging.info("\n--- Random Forest ---")
    rf_model = load_model(CALIBRATED_RF_MODEL_PATH)
    p_rf = rf_model.predict_proba(X_test)[:, 1]
    thrs_rf, costs_rf, details_rf = sweep_costs(y_test, p_rf, n=201)
    best_idx_rf = int(np.argmin(costs_rf))
    best_thr_rf = float(thrs_rf[best_idx_rf])
    best_tp_rf, best_fp_rf, best_fn_rf, best_tn_rf = details_rf[best_idx_rf]

    plt.figure()
    plt.plot(thrs_rf, costs_rf)
    plt.title("Krzywa kosztu vs próg - Las Losowy")
    plt.xlabel("Próg")
    plt.ylabel("Koszt (niżej lepiej)")
    plt.savefig(f"{PLOTS_PATH}/cost_curve_rf.png")
    plt.close()

    logging.info(f"Najlepszy próg RF: {best_thr_rf:.4f}")
    logging.info(f"TP, FP, FN, TN: {best_tp_rf}, {best_fp_rf}, {best_fn_rf}, {best_tn_rf}")
    accept_rate_rf = (best_tn_rf + best_fn_rf) / len(y_test)
    logging.info(f"Stopa akceptacji przy najlepszym progu: {accept_rate_rf:.4f}")
    
    logging.info("=" * 50)
    
    return best_thr_lr, best_thr_rf


if __name__ == "__main__":
    main()