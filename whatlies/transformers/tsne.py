import numpy as np
from sklearn.manifold import TSNE

from whatlies import EmbeddingSet
from whatlies.transformers.common import embset_to_X, new_embedding_dict


class Tsne:
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

    emb.transform(Tsne(3)).plot_interactive_matrix('tsne_0', 'tsne_1', 'tsne_2')
    ```
    """

    def __init__(self, n_components=2, **kwargs):
        self.is_fitted = False
        self.n_components = n_components
        self.kwargs = kwargs
        self.tfm = TSNE(n_components=n_components, **kwargs)

    def __call__(self, embset):
        if not self.is_fitted:
            self.fit(embset)
        return self.transform(embset)

    def fit(self, embset):
        names, X = embset_to_X(embset=embset)
        self.tfm.fit(X)
        self.is_fitted = True

    def transform(self, embset):
        names, X = embset_to_X(embset=embset)
        new_vecs = self.tfm.fit_transform(X)
        names_out = names + [f"tsne_{i}" for i in range(self.n_components)]
        vectors_out = np.concatenate([new_vecs, np.eye(self.n_components)])
        new_dict = new_embedding_dict(names_out, vectors_out, embset)
        return EmbeddingSet(new_dict, name=f"{embset.name}.tsne_{self.n_components}()")
