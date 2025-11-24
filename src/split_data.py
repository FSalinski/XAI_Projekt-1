'''
Script for splitting data into training, validation, and test sets
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from constants import RANDOM_STATE, TEST_SIZE

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("=" * 50)
    logging.info("Starting data splitting")

    # Load data
    df = pd.read_csv('data/zbiór_5.csv')
    X = df.drop(columns=['default'])
    y = df['default']

    # Split data into training+validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    logging.info("Data split into training and testing sets")

    # Save splits to CSV files
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df.to_csv('data/train.csv', index=False)
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv('data/test.csv', index=False)
    logging.info("Saved training and testing sets to CSV files")
    logging.info("=" * 50)

if __name__ == "__main__":
    main()
