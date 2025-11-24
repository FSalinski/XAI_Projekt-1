import os

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RANDOM_STATE = 2137

TEST_SIZE = 0.3
TARGET_COLUMN = 'default'

MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
PLOTS_PATH = os.path.join(PROJECT_ROOT, 'plots')
TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'train.csv')
TEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'test.csv')
TUNED_LR_MODEL_PATH = os.path.join(MODELS_PATH, 'tuned_logistic_regression.pkl')
TUNED_RF_MODEL_PATH = os.path.join(MODELS_PATH, 'tuned_random_forest.pkl')
CALIBRATED_LR_MODEL_PATH = os.path.join(MODELS_PATH, 'calibrated_logistic_regression.pkl')
CALIBRATED_RF_MODEL_PATH = os.path.join(MODELS_PATH, 'calibrated_random_forest.pkl')