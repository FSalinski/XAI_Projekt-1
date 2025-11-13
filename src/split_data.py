'''
Script for splitting data into training, validation, and test sets
'''

import pandas as pd
from sklearn.model_selection import train_test_split
import logging

RANDOM_STATE = 2137

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("=" * 50)
    logging.info("Starting data splitting")

    # Load data
    df = pd.read_csv('data/zbiór_5.csv')
    X = df.drop(columns=['default'])
    y = df['default']

    # Split data into training+validation and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    # Further split training+validation into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_temp)
    logging.info("Data split into training, validation, and testing sets")

    # Save splits to CSV files
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df.to_csv('data/train.csv', index=False)
    val_df = pd.concat([X_val, y_val], axis=1)
    val_df.to_csv('data/val.csv', index=False)
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv('data/test.csv', index=False)
    logging.info("Saved training, validation, and testing sets to CSV files")
    logging.info("=" * 50)

if __name__ == "__main__":
    main()
