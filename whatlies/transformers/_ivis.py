from ._transformer import SklearnTransformer

from ivis import Ivis as IVIS


class Ivis(SklearnTransformer):
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
    from whatlies.language import SpacyLanguage
    from whatlies.transformers import Ivis

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "bluee", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]
    emb.transform(Ivis(3)).plot_interactive_matrix(0, 1, 2)
    ```
    """

    def __init__(self, n_components=2, **kwargs):
        # Our API keeps referring to `n_components` to keep things standard but IVIS calls it
        # `embedding_dims` internally.
        super().__init__(
            IVIS, f"ivis_{n_components}", embedding_dims=n_components, **kwargs
        )
