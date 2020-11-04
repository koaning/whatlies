from whatlies import Embedding, EmbeddingSet
from whatlies.language._common import SklearnTransformerMixin
from sentence_transformers import SentenceTransformer


class SentenceTFMLanguage(SklearnTransformerMixin):
    """
    This class provides the abitilty to load and use the encoding strategies found in the [sentence transformers](https://www.sbert.net/index.html) library.

    A full list of pretrained embeddings can be found [here](https://www.sbert.net/docs/pretrained_models.html).

    Arguments:
        name: name of the model to load, be sure that it's downloaded or trained beforehand

    **Usage**:

    ```python
    > from whatlies.language import SentenceTFMLanguage
    > lang = SentenceTFMLanguage('distilbert-base-nli-stsb-mean-tokens')
    > lang['python is great']
    > lang[['python is a language', 'python is a snake', 'dogs are cool animals']]
    ```
    """

    def __init__(self, name):
        self.name = name
        try:
            self.model = SentenceTransformer(name)
        except OSError:
            raise ValueError(
                f"Make sure that the name (got {name}) is found on this website:\nhttps://www.sbert.net/docs/pretrained_models.html"
            )

    def __getitem__(self, query):
        if isinstance(query, str):
            return Embedding(query, vector=self.model.encode(query))
        else:
            return EmbeddingSet(*[self[tok] for tok in query])
