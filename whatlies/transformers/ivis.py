from whatlies import EmbeddingSet
from whatlies.transformers.common import embset_to_X, new_embedding_dict

from ivis import Ivis as IVIS
import numpy as np


class Ivis:
    """
    This transformer scales all the vectors in an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]
    by means of Ivis algorithm. We're using the implementation found
    [here](https://github.com/beringresearch/ivis).

    Arguments:
        n_components: the number of compoments to create/add
        kwargs: keyword arguments passed to the [Ivis implementation](https://bering-ivis.readthedocs.io/en/latest/hyperparameters.html)

    Usage:

    ```python
    from whatlies.language import GensimLanguage
    from whatlies.transformers import Ivis

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = GensimLanguage("wordvectors.kv")
    emb = lang[words]
    emb.transform(Ivis(3)).plot_interactive_matrix('ivis_0', 'ivis_1', 'ivis_2')
    ```
    """

    def __init__(self, n_components=2, **kwargs):
        self.is_fitted = False
        self.n_components = n_components
        self.kwargs = kwargs
        self.kwargs["verbose"] = 0
        self.tfm = IVIS(embedding_dims=self.n_components, **self.kwargs)

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
        names_out = names + [f"ivis_{i}" for i in range(self.n_components)]
        vectors_out = np.concatenate([new_vecs, np.eye(self.n_components)])
        new_dict = new_embedding_dict(names_out, vectors_out, embset)
        return EmbeddingSet(new_dict, name=f"{embset.name}.ivis_{self.n_components}()")
