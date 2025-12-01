import os

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#---------- GENERAL CONSTANTS ----------

RANDOM_STATE = 2137 # random seed used throughout the whole project for reproducibility

TEST_SIZE = 0.3 # test set size for the initial train-test split (for calibration the test set is further split in two into validation and test)
TARGET_COLUMN = 'default' # name of the target column in the dataset
UNIQUE_VALUES_THRESHOLD = 0.95 # threshold for dropping columns with a high percentage of identical values during manual preprocessing

#---------- PATH CONSTANTS ----------
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
PLOTS_PATH = os.path.join(PROJECT_ROOT, 'plots')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
ORIGINAL_DATASET_PATH = os.path.join(DATA_PATH, 'zbiór_5.csv')
PREPROCESSED_DATASET_PATH = os.path.join(DATA_PATH, 'zbiór_5_preprocessed.csv')
TRAIN_PATH = os.path.join(DATA_PATH, 'train.csv')
TEST_PATH = os.path.join(DATA_PATH, 'test.csv')
TUNED_LR_MODEL_PATH = os.path.join(MODELS_PATH, 'tuned_logistic_regression.pkl')
TUNED_RF_MODEL_PATH = os.path.join(MODELS_PATH, 'tuned_random_forest.pkl')
CALIBRATED_LR_MODEL_PATH = os.path.join(MODELS_PATH, 'calibrated_logistic_regression.pkl')
CALIBRATED_RF_MODEL_PATH = os.path.join(MODELS_PATH, 'calibrated_random_forest.pkl')