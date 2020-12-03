import warnings
from pathlib import Path
from typing import Union, List

import numpy as np
from bpemb import BPEmb
from sklearn.metrics import pairwise_distances

from whatlies import Embedding, EmbeddingSet
from whatlies.language._common import SklearnTransformerMixin


class BytePairLanguage(SklearnTransformerMixin):
    """
    This object is used to lazily fetch [Embedding][whatlies.embedding.Embedding]s or
    [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]s from a Byte-Pair Encoding
    backend. This object is meant for retreival, not plotting.

    This language represents token-free pre-trained subword embeddings. Originally created by
    Benjamin Heinzerling and Michael Strube.

    Important:
        These vectors will auto-download by the [BPEmb package](https://nlp.h-its.org/bpemb/).
        You can also specify "multi" to download multi language embeddings. A full list of available
        languages can be found [here](https://nlp.h-its.org/bpemb). The article that
        belongs to this work can be found [here](http://www.lrec-conf.org/proceedings/lrec2018/pdf/1049.pdf)
        Recognition should be given to Benjamin Heinzerling and Michael Strube for making these available.
        The availability of vocabulary size as well as dimensionality can be varified
        on the project website. See [here](https://nlp.h-its.org/bpemb/en/) for an
        example link in English. Please credit the original authors if you use their work.

    Warning:
        This class used to be called `BytePairLang`.

    Arguments:
        lang: name of the model to load
        vs: vocabulary size of the byte pair model
        dim: the embedding dimensionality
        cache_dir: The folder in which downloaded BPEmb files will be cached

    Typically the vocabulary size given from this backend can be of size 1000,
    3000, 5000, 10000, 25000, 50000, 100000 or 200000. The available dimensionality
    of the embbeddings typically are 25, 50, 100, 200 and 300.

    **Usage**:

    ```python
    > from whatlies.language import BytePairLanguage
    > lang = BytePairLanguage(lang="en")
    > lang['python']
    > lang = BytePairLanguage(lang="multi")
    > lang[['hund', 'hond', 'dog']]
    ```
    """

    def __init__(
        self, lang, vs=10000, dim=100, cache_dir=Path.home() / Path(".cache/bpemb")
    ):
        self.lang = lang
        self.vs = vs
        self.dim = dim
        self.cache_dir = cache_dir
        self.module = BPEmb(lang=lang, vs=vs, dim=dim, cache_dir=cache_dir)

    def __getitem__(self, item):
        """
        Retreive a single embedding or a set of embeddings. If an embedding contains multiple
        sub-tokens then we'll average them before retreival.

        Arguments:
            item: single string or list of strings

        **Usage**
        ```python
        > lang = BytePairLanguage(lang="en")
        > lang['python']
        > lang[['python', 'snake']]
        > lang[['nobody expects', 'the spanish inquisition']]
        ```
        """
        if isinstance(item, str):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                return Embedding(item, self.module.embed(item).mean(axis=0))
        if isinstance(item, list):
            return EmbeddingSet(*[self[i] for i in item])
        raise ValueError(f"Item must be list of string got {item}.")

    def _prepare_queries(self, lower):
        queries = [w for w in self.module.emb.vocab.keys()]
        if lower:
            queries = [w for w in queries if w.lower() == w]
        return queries

    def _calculate_distances(self, emb, queries, metric):
        vec = emb.vector
        vector_matrix = np.array([self[w].vector for w in queries])
        # there are NaNs returned, good to investigate later why that might be
        vector_matrix = np.array(
            [np.zeros(v.shape) if np.any(np.isnan(v)) else v for v in vector_matrix]
        )
        return pairwise_distances(vector_matrix, vec.reshape(1, -1), metric=metric)

    def score_similar(
        self,
        emb: Union[str, Embedding],
        n: int = 10,
        metric="cosine",
        lower=False,
    ) -> List:
        """
        Retreive a list of (Embedding, score) tuples that are the most similar to the passed query.

        Arguments:
            emb: query to use
            n: the number of items you'd like to see returned
            metric: metric to use to calculate distance, must be scipy or sklearn compatible
            lower: only fetch lower case tokens

        Returns:
            An list of ([Embedding][whatlies.embedding.Embedding], score) tuples.
        """
        if isinstance(emb, str):
            emb = self[emb]

        queries = self._prepare_queries(lower=lower)
        distances = self._calculate_distances(emb=emb, queries=queries, metric=metric)
        by_similarity = sorted(zip(queries, distances), key=lambda z: z[1])

        if len(queries) < n:
            warnings.warn(
                f"We could only find {len(queries)} feasible words. Consider changing `top_n` or `lower`",
                UserWarning,
            )

        return [(self[q], float(d)) for q, d in by_similarity[:n]]

    def embset_similar(
        self,
        emb: Union[str, Embedding],
        n: int = 10,
        lower=False,
        metric="cosine",
    ) -> EmbeddingSet:
        """
        Retreive an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] that are the most similar to the passed query.

        Arguments:
            emb: query to use
            n: the number of items you'd like to see returned
            metric: metric to use to calculate distance, must be scipy or sklearn compatible
            lower: only fetch lower case tokens

        Important:
            This method is incredibly slow at the moment without a good `top_n` setting due to
            [this bug](https://github.com/facebookresearch/fastText/issues/1040).

        Returns:
            An [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] containing the similar embeddings.
        """
        embs = [
            w[0] for w in self.score_similar(emb=emb, n=n, lower=lower, metric=metric)
        ]
        return EmbeddingSet({w.name: w for w in embs})
