from openTSNE import TSNE

from ._transformer import SklearnTransformer


class OpenTsne(SklearnTransformer):
    """
    This transformer transformers all vectors in an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]
    by means of tsne. This implementation used
    [open-tsne](https://opentsne.readthedocs.io/en/latest/tsne_algorithm.html).

    Important:
        OpenTSNE is a faster variant of TSNE but it only allows for <2 components.
        You may also notice that it is relatively slow. This unfortunately is a fact of TSNE.

        This embedding transformation might require you to manually install extra dependencies
        unless you installed via either;

        ```
        pip install whatlies[opentsne]
        pip install whatlies[all]
        ```

    Arguments:
        n_components: the number of compoments to create/add
        kwargs: keyword arguments passed to the OpenTsne implementation, includes things like `perplexity` [link](https://opentsne.readthedocs.io/en/latest/api/index.html)

    Usage:

    ```python
    from whatlies.language import SpacyLanguage
    from whatlies.transformers import OpenTsne

    words = ["prince", "princess", "nurse", "doctor", "banker", "man", "woman",
             "cousin", "neice", "king", "queen", "dude", "guy", "gal", "fire",
             "dog", "cat", "mouse", "red", "blue", "green", "yellow", "water",
             "person", "family", "brother", "sister"]

    lang = SpacyLanguage("en_core_web_md")
    emb = lang[words]

    emb.transform(OpenTsne(2)).plot_interactive_matrix()
    ```
    """

    def __init__(self, n_components=2, **kwargs):
        super().__init__(
            TSNE, f"opentsne_{n_components}", n_components=n_components, **kwargs
        )
