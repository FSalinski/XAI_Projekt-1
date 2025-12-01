'''
Manual preprocessing (before train test split)
'''

import logging
import pandas as pd
from constants import UNIQUE_VALUES_THRESHOLD, ORIGINAL_DATASET_PATH, PREPROCESSED_DATASET_PATH

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load data
    df = pd.read_csv(ORIGINAL_DATASET_PATH)

    # Drop columns with > UNIQUE_VALUES_THRESHOLD identical values
    cols_to_drop = []
    for col in df.columns:
        most_frequent_value = df[col].value_counts(normalize=True).iloc[0]
        if most_frequent_value > UNIQUE_VALUES_THRESHOLD:
            cols_to_drop.append(col)
    
    df = df.drop(columns=cols_to_drop)
    logging.info(f"Number of columns dropped: {len(cols_to_drop)}")
    logging.info(f"Dropped columns due to high identical value percentage: {cols_to_drop}")

    # Save the preprocessed dataset
    df.to_csv(PREPROCESSED_DATASET_PATH, index=False)
    logging.info(f"Preprocessed dataset saved to {PREPROCESSED_DATASET_PATH}")


if __name__ == "__main__":
    main()
