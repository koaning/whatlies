from sklearn.utils import check_array
import numpy as np


class SklearnTransformerMixin:

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = check_array(X, dtype="object")
        X = X.reshape(-1)
        return np.array([self[x].vector for x in X])

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
