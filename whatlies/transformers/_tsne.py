from sklearn.manifold import TSNE

from ._transformer import SklearnTransformer
from whatlies import EmbeddingSet
from whatlies.transformers._common import new_embedding_dict


class Tsne(SklearnTransformer):
    """
    This transformer transformers all vectors in an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]
    by means of tsne. This implementation uses
    [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE).

    Important:
        TSNE does not allow you to train a transformation and re-use it. It must retrain every time it sees data.
        You may also notice that it is relatively slow. This unfortunately is a fact of life.

    Arguments:
        n_components: the number of compoments to create/add
        kwargs: keyword arguments passed to the Tsne implementation, includes things like `perplexity` [link](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE)

    Usage:

    ```python
    from whatlies.language import SpacyLanguage
    from whatlies.transformers import Tsne

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "blue", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(Tsne(3)).plot_interactive_matrix(0, 1, 2)
    ```
    """

    def __init__(self, n_components=2, **kwargs):
        self.n_components = n_components
        super().__init__(
            TSNE, f"tsne_{n_components}", n_components=n_components, **kwargs
        )

    def transform(self, embset):
        names, X = embset.to_names_X()
        # We are re-writing the transform method here because TSNE cannot .fit().transform().
        # Check the docs here: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE
        new_vecs = self.tfm.fit_transform(X)
        new_dict = new_embedding_dict(names, new_vecs, embset)
        return EmbeddingSet(new_dict, name=f"{embset.name}.tsne({self.n_components})")
