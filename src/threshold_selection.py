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
    p = load_model(CALIBRATED_LR_MODEL_PATH).predict_proba(X_test)[:, 1]
    thrs, costs, details = sweep_costs(y_test, p, n=201)
    best_idx = int(np.argmin(costs))
    best_thr = float(thrs[best_idx])
    best_tp, best_fp, best_fn, best_tn = details[best_idx]

    plt.figure()
    plt.plot(thrs, costs)
    plt.title("Krzywa kosztu vs próg")
    plt.xlabel("Próg")
    plt.ylabel("Koszt (niżej lepiej)")
    # save plot
    plt.savefig(f"{PLOTS_PATH}/cost_curve_lr.png")
    plt.close()

    logging.info(f"Najlepszy próg: {best_thr}")
    logging.info(f"TP, FP, FN, TN: {best_tp}, {best_fp}, {best_fn}, {best_tn}")
    accept_rate = (best_tn + best_fn) / len(y_test)
    logging.info(f"Stopa akceptacji przy najlepszym progu: {accept_rate:.4f}")


if __name__ == "__main__":
    main()