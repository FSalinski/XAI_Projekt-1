import pandas as pd
import numpy as np
import pickle
from sklearn.calibration import calibration_curve
from sklearn.metrics import recall_score, roc_auc_score, brier_score_loss, log_loss
from constants import TRAIN_PATH, TEST_PATH, TARGET_COLUMN

# ---------- MODEL UTILS ----------
def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return recall, roc_auc

# ---------- DATA LOADING UTILS ----------
def load_train_test_data(train_path=TRAIN_PATH, test_path=TEST_PATH, target_column=TARGET_COLUMN):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    return X_train, X_test, y_train, y_test


# ---------- CALIBRATION UTILS ----------
def calibration_metrics_dict(y_true, p):
    return {
        "AUC": roc_auc_score(y_true, p),
        "Recall": recall_score(y_true, (p >= 0.5).astype(int)),
        "Brier": brier_score_loss(y_true, p),
        "LogLoss": log_loss(y_true, p)
    }

def expected_calibration_error(y_true, p, n_bins=10):
    quantiles = np.linspace(0, 1, n_bins+1)
    bins = np.quantile(p, quantiles)
    bins = np.unique(bins)
    
    if len(bins) == 1:
        return float(np.abs(y_true.mean() - p.mean()))
    
    idx = np.digitize(p, bins[1:-1], right=True)
    ece = 0.0
    
    for b in range(len(bins)-1):
        mask = idx == b
        if np.sum(mask) == 0:
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        w = np.mean(mask)
        ece += w * np.abs(acc - conf)
    
    return float(ece)

def reliability_diagram(ax, y_true, p, label="model"):
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0,0.3], [0,0.3], "--", color="gray")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Rate of positive outcomes")
    ax.set_title("Reliability diagram")
    ax.grid(alpha=0.3)