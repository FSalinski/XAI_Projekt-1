import os

# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RANDOM_STATE = 2137

TEST_SIZE = 0.3
TARGET_COLUMN = 'default'

TRAIN_PATH = os.path.join(PROJECT_ROOT, 'data', 'train.csv')
TEST_PATH = os.path.join(PROJECT_ROOT, 'data', 'test.csv')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
TUNED_LR_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'tuned_logistic_regression.pkl')
TUNED_RF_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'tuned_random_forest.pkl')
CALIBRATED_LR_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'calibrated_logistic_regression.pkl')
CALIBRATED_RF_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'calibrated_random_forest.pkl')