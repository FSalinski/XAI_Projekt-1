import pandas as pd
import numpy as np

def find_correlated_pairs(corr, threshold=0.75):
    correlated_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > threshold and i != j:
                correlated_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    return correlated_pairs

def remove_correlated_features(df, correlated_pairs):
    features_to_remove = []
    for feature1, feature2, _ in correlated_pairs:
        if feature2 not in features_to_remove and feature1 not in features_to_remove:
            features_to_remove.append(feature2)
    return df.drop(columns=features_to_remove)

def remove_all_correlated_features(df, threshold=0.75):
    while True:
        corr = df.corr(numeric_only=True)
        correlated_pairs = find_correlated_pairs(corr, threshold)
        if not correlated_pairs: # if no more correlated pairs, break
            break
        df = remove_correlated_features(df, correlated_pairs)
    return df