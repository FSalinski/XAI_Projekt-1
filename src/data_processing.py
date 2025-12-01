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
        if self.alpha == 0:
            return self
        X_ = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.columns_ = X_.columns
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for col in self.columns_:
            self.lower_bounds_[col] = np.nanquantile(X_[col], self.lower)
            self.upper_bounds_[col] = np.nanquantile(X_[col], self.upper)
        return self

    def transform(self, X):
        if self.alpha == 0:
            return X
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_", "columns_"])
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.columns_)
        for col in self.columns_:
            low = self.lower_bounds_[col]
            up = self.upper_bounds_[col]
            X_df[col] = X_df[col].clip(lower=low, upper=up)
        return X_df if isinstance(X, pd.DataFrame) else X_df.to_numpy()

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.columns_, dtype=object)
        return np.array(input_features, dtype=object)
    
class PairwiseCorrelatedFeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.features_to_remove_ = []
        self.corr_matrix_ = None

    def fit(self, X, y=None):
        if self.threshold >= 1.0:
            return self
        X_ = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.corr_matrix_ = X_.corr(numeric_only=True)
        while True:
            correlated_pairs = self.find_correlated_pairs(self.corr_matrix_, self.threshold)
            if not correlated_pairs: # if no more correlated pairs, break
                break
            for feature1, feature2, _ in correlated_pairs:
                if feature2 not in self.features_to_remove_ and feature1 not in self.features_to_remove_:
                    self.features_to_remove_.append(feature2)
                    self.corr_matrix_ = self.corr_matrix_.drop(index=feature2, columns=feature2)
        return self

    def transform(self, X):
        if self.threshold >= 1.0:
            return X
        check_is_fitted(self, ["features_to_remove_"])
        X_ = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return X_.drop(columns=self.features_to_remove_)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if self.corr_matrix_ is not None:
                return self.corr_matrix_.columns.to_numpy()
            else:
                raise ValueError("Estimator not fitted or input features not provided")
        
        return np.array([f for f in input_features if f not in self.features_to_remove_], dtype=object)
    
    def find_correlated_pairs(self, corr, threshold=0.75):
        correlated_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > threshold and i != j:
                    correlated_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        return correlated_pairs


def data_processing_pipeline_lr(X, alpha=0.05, corr_threshold=0.75, imputer_strategy='median', add_indicator=True):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('replace_inf', FunctionTransformer(replace_inf)),
        ('imputer', SimpleImputer(strategy=imputer_strategy, add_indicator=add_indicator)),
        ('quantile_clipper', QuantileClipper(alpha=alpha)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False
    )
    
    # Enable pandas output to preserve feature names
    preprocessor.set_output(transform="pandas")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('correlated_feature_remover', PairwiseCorrelatedFeatureRemover(threshold=corr_threshold))])

    return pipeline


def data_processing_pipeline_rf(X, alpha=0.05, corr_threshold=0.75, imputer_strategy='median', add_indicator=True):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('replace_inf', FunctionTransformer(replace_inf)),
        ('imputer', SimpleImputer(strategy=imputer_strategy, add_indicator=add_indicator)),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False
    )
    
    # Enable pandas output to preserve feature names
    preprocessor.set_output(transform="pandas")

    return preprocessor
