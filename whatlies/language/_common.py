import os
import sys

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin


class SklearnTransformerMixin(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """
        Will fit the language model such that it is ready for use in scikit learn.
        Check out the [guide](https://koaning.github.io/whatlies/tutorial/languages/#scikit-learn) for more details.
        """
        if not np.array(X).dtype.type is np.str_:
            raise ValueError("You must give this preprocessor text as input.")
        self.fitted_ = True
        return self

    def partial_fit(self, X, y=None):
        """
        No-op.
        """
        if not np.array(X).dtype.type is np.str_:
            raise ValueError("You must give this preprocessor text as input.")
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Will apply the transformer as if it is a scikit-learn text feature extractor.
        Check out the [guide](https://koaning.github.io/whatlies/tutorial/languages/#scikit-learn) for more details.
        """
        check_is_fitted(self, "fitted_")
        if not np.array(X).dtype.type is np.str_:
            raise ValueError("You must give this preprocessor text as input.")
        return np.array([np.nan_to_num(self[x].vector) for x in X])


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stderr
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stdout
