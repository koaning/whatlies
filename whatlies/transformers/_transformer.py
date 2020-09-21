from abc import ABC, abstractmethod

from whatlies import EmbeddingSet
from whatlies.transformers._common import new_embedding_dict


class Transformer(ABC):
    """
    This is the abstract base class for all the transformers. Each subclass of
    this class should at least implement the `fit` and `transform` methods.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.is_fitted = False

    def __call__(self, embset: EmbeddingSet) -> EmbeddingSet:
        if not self.is_fitted:
            self.fit(embset)
        return self.transform(embset)

    @abstractmethod
    def fit(self, embset: EmbeddingSet) -> "Transformer":
        """
        Fit the transformer on the given `EmbeddingSet` instance (if neccessary). This method should
        set the `is_fitted` flag to `True`.

        Arguments:
            embset: an `EmbeddingSet` instance used for fitting the transformer.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, embset: EmbeddingSet) -> EmbeddingSet:
        """
        Transform the given `EmbeddingSet` instance.

        Arguments:
            embset: an `EmbeddingSet` instance to be transformed.
        """
        raise NotImplementedError


class SklearnTransformer(Transformer):
    """
    This class is a wrapper around scikit-learn. Since many of our transformers follow
    the scikit-learn API we might be able to save a whole lot of code this way.
    """

    def __init__(self, tfm, name, *args, **kwargs):
        super().__init__()
        self.tfm = tfm(*args, **kwargs)
        self.name = name

    def fit(self, embset: EmbeddingSet) -> "SklearnTransformer":
        if not self.is_fitted:
            # This is a bit of an anti-pattern. You should not need to `self.tfm`. However, there are
            # some packages like OpenTSNE that return a different kind of estimator once an estimator has been fitted.
            self.tfm = self.tfm.fit(embset.to_X())
        self.is_fitted = True
        return self

    def transform(self, embset: EmbeddingSet) -> EmbeddingSet:
        names, X = embset.to_names_X()
        if not self.is_fitted:
            self.tfm.fit(X)
        new_vecs = self.tfm.transform(X)
        new_dict = new_embedding_dict(names, new_vecs, embset)
        return EmbeddingSet(new_dict, name=f"{embset.name}.{self.name}()")
