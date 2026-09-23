"""Importable production version of the Chapter 18 transformer.

Identical to the class written in lessons/ch18_custom_and_production.py.
A fitted model saved with joblib must be able to import its classes in a
fresh Python process, so the saved pipeline uses this copy.
"""
import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    '''Clip each column to percentiles learned from the training data.'''

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower                       # stored unchanged
        self.upper = upper

    def fit(self, X, y=None):
        X = validate_data(self, X)
        self.lower_ = np.quantile(X, self.lower, axis=0)    # learned -> underscore
        self.upper_ = np.quantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return np.clip(X, self.lower_, self.upper_)
