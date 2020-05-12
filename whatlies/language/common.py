from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class SklearnTransformerMixin(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """
        Will fit the language model such that it is ready for use in scikit learn.
        Check out the [guide](https://rasahq.github.io/whatlies/tutorial/languages/#scikit-learn) for more details.
        """
        if not np.array(X).dtype.type is np.str_:
            raise ValueError("You must give this preprocessor text as input.")
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Will apply the transformer as if it is a scikit-learn text feature extractor.
        Check out the [guide](https://rasahq.github.io/whatlies/tutorial/languages/#scikit-learn) for more details.
        """
        check_is_fitted(self, "fitted_")
        if not np.array(X).dtype.type is np.str_:
            raise ValueError("You must give this preprocessor text as input.")
        return np.array([self[x].vector for x in X])
