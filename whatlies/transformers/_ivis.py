from whatlies.transformers import Transformer
from whatlies import EmbeddingSet
from whatlies.transformers._common import new_embedding_dict

from ivis import Ivis as IVIS


class Ivis(Transformer):
    """
    This transformer scales all the vectors in an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]
    by means of Ivis algorithm. We're using the implementation found
    [here](https://github.com/beringresearch/ivis).

    Important:
        This language backend might require you to manually install extra dependencies
        unless you installed via either;

        ```
        pip install whatlies[ivis]
        pip install whatlies[all]
        ```

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
    emb.transform(Ivis(3)).plot_interactive_matrix(0, 1, 2)
    ```
    """

    def __init__(self, n_components=2, **kwargs):
        super().__init__()
        self.n_components = n_components
        self.kwargs = kwargs
        self.kwargs["verbose"] = 0
        self.tfm = IVIS(embedding_dims=self.n_components, **self.kwargs)

    def fit(self, embset):
        names, X = embset.to_names_X()
        self.tfm.fit(X)
        self.is_fitted = True
        return self

    def transform(self, embset):
        names, X = embset.to_names_X()
        new_vecs = self.tfm.transform(X)
        new_dict = new_embedding_dict(names, new_vecs, embset)
        return EmbeddingSet(new_dict, name=f"{embset.name}.ivis_{self.n_components}()")
