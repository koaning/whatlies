from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class SklearnTransformerMixin(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = check_array(X, dtype="object")
        if np.array(X).dtype.type is np.string_:
            raise ValueError("You must give this preprocessor text as input.")
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "fitted_")
        if np.array(X).dtype.type is np.string_:
            raise ValueError("You must give this preprocessor text as input.")
        X = check_array(X, dtype="object")
        X = X.reshape(-1)
        return np.array([self[x].vector for x in X])
