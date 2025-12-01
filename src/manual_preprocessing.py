'''
Manual preprocessing (before train test split)
'''

import logging
import pandas as pd
from constants import UNIQUE_VALUES_THRESHOLD, ZEROES_TO_NAN_THRESHOLD, ORIGINAL_DATASET_PATH, PREPROCESSED_DATASET_PATH

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load data
    df = pd.read_csv(ORIGINAL_DATASET_PATH)

    # Drop columns with > UNIQUE_VALUES_THRESHOLD identical values
    # For other columns replace > ZEROES_TO_NAN_THRESHOLD zeroes with NaN
    cols_to_drop = []
    zeroes_to_nan_cols = []
    for col in df.columns:
        most_frequent_value = df[col].value_counts(normalize=True).iloc[0]
        if most_frequent_value > UNIQUE_VALUES_THRESHOLD:
            cols_to_drop.append(col)
        elif col != 'default':  # avoid replacing zeroes in target column
            zero_fraction = (df[col] == 0).mean()
            if zero_fraction > ZEROES_TO_NAN_THRESHOLD:
                df.loc[df[col] == 0, col] = pd.NA
                zeroes_to_nan_cols.append(col)
    
    df = df.drop(columns=cols_to_drop)
    logging.info(f"Dropped columns due to high (over {UNIQUE_VALUES_THRESHOLD*100}%) identical value percentage: {cols_to_drop}")
    logging.info(f"Number of columns dropped: {len(cols_to_drop)}")
    logging.info(f"Replaced zeroes with NaN in columns: {zeroes_to_nan_cols}, since they had more than {ZEROES_TO_NAN_THRESHOLD*100}% zeroes")

    # Save the preprocessed dataset
    df.to_csv(PREPROCESSED_DATASET_PATH, index=False)
    logging.info(f"Preprocessed dataset saved to {PREPROCESSED_DATASET_PATH}")


if __name__ == "__main__":
    main()
