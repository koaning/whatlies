from abc import ABC, abstractmethod

from whatlies import EmbeddingSet


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
