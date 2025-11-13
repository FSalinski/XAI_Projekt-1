from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

def replace_inf(X):
    X = X.copy()
    for col in X.select_dtypes(include=[np.number]).columns:
        maximum = X[col].filter(X[col] < np.inf).max()
        X[col] = X[col].replace([np.inf], maximum)
    return X

class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.lower = alpha
        self.upper = 1-alpha
        assert 0 <= self.lower < self.upper <= 1, "lower < upper and both in [0,1]"

    def fit(self, X, y=None):
        X_ = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.columns_ = X_.columns
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in self.columns_:
            self.lower_bounds_[col] = np.nanquantile(X_[col], self.lower)
            self.upper_bounds_[col] = np.nanquantile(X_[col], self.upper)
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_", "columns_"])
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns_)
        for col in self.columns_:
            low = self.lower_bounds_[col]
            up = self.upper_bounds_[col]
            X_df[col] = X_df[col].clip(lower=low, upper=up)
        return X_df if isinstance(X, pd.DataFrame) else X_df.to_numpy()

def baseline_data_processing_pipeline(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('replace_inf', FunctionTransformer(replace_inf)),
        ('imputer', SimpleImputer(strategy='median')),
        ('quantile_clipper', QuantileClipper(alpha=0.05)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor
