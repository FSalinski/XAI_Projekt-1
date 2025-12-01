import os

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#---------- GENERAL CONSTANTS ----------

RANDOM_STATE = 2137 # random seed used throughout the whole project for reproducibility

TEST_SIZE = 0.3 # test set size for the initial train-test split (for calibration the test set is further split in two into validation and test)
TARGET_COLUMN = 'default' # name of the target column in the dataset
UNIQUE_VALUES_THRESHOLD = 0.95 # threshold for dropping columns with a high percentage of identical values during manual preprocessing
ZEROES_TO_NAN_THRESHOLD = 0.75 # threshold for replacing columns with a high percentage of zeroes with NaN during manual preprocessing
MAX_FEATURES = 100 # number of features to select during feature selection using RFE

#---------- COST CONSTANTS ----------
COST_TP = 0.0    # poprawnie nie udzielony kredyt niewypłacalnemu klientowi
COST_FN =  100000.0    # koszt udzielenia kredytu niewypłacalnemu klientowi
COST_FP = 0.0
COST_TN = -10000.0   # poprawnie udzielony kredyt wypłacalnemu klientowi

#---------- PATH CONSTANTS ----------
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
PLOTS_PATH = os.path.join(PROJECT_ROOT, 'plots')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
ORIGINAL_DATASET_PATH = os.path.join(DATA_PATH, 'zbiór_5.csv')
PREPROCESSED_DATASET_PATH = os.path.join(DATA_PATH, 'zbiór_5_preprocessed.csv')
TRAIN_PATH = os.path.join(DATA_PATH, 'train.csv')
TEST_PATH = os.path.join(DATA_PATH, 'test.csv')
TRAIN_TRIMMED_PATH = os.path.join(DATA_PATH, 'train_trimmed.csv')
TEST_TRIMMED_PATH = os.path.join(DATA_PATH, 'test_trimmed.csv')
TUNED_LR_MODEL_PATH = os.path.join(MODELS_PATH, 'tuned_logistic_regression.pkl')
TUNED_RF_MODEL_PATH = os.path.join(MODELS_PATH, 'tuned_random_forest.pkl')
CALIBRATED_LR_MODEL_PATH = os.path.join(MODELS_PATH, 'calibrated_logistic_regression.pkl')
CALIBRATED_RF_MODEL_PATH = os.path.join(MODELS_PATH, 'calibrated_random_forest.pkl')