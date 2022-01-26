import warnings
from typing import Union, List

import spacy
from spacy.language import Language
import numpy as np
from sklearn.metrics import pairwise_distances

from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet
from whatlies.language._common import SklearnTransformerMixin


class SpacyLanguage(SklearnTransformerMixin):
    """
    This object is used to lazily fetch [Embedding][whatlies.embedding.Embedding]s or
    [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]s from a spaCy language
    backend. This object is meant for retreival, not plotting.

    Arguments:
        nlp: name of the model to load, be sure that it's downloaded beforehand

    Important:
        This language backend might require you to manually install extra dependencies
        unless you installed via either;

        ```
        pip install whatlies[spacy]
        pip install whatlies[all]
        ```

    **Usage**:

    ```python
    > lang = SpacyLanguage("en_core_web_md")
    > lang['python']
    > lang[['python', 'snake', 'dog']]
    ```
    """

    def __init__(self, nlp: Union[str, Language]):
        self.nlp = nlp
        if isinstance(nlp, str):
            self.model = spacy.load(nlp)
        elif isinstance(nlp, Language):
            self.model = nlp
        else:
            raise ValueError(
                "Language must be started with `str` or spaCy-language object."
            )
        # Remove the lexeme prob table if it exists and is empty
        if (
            hasattr(self.model.vocab, "lookups_extra")
            and self.model.vocab.lookups_extra.has_table("lexeme_prob")
            and len(self.model.vocab.lookups_extra.get_table("lexeme_prob")) == 0
        ):
            self.model.vocab.lookups_extra.remove_table("lexeme_prob")

    def __getitem__(
        self, query: Union[str, List[str]]
    ) -> Union[Embedding, EmbeddingSet]:
        """
        Retreive a single embedding or a set of embeddings. Depending on the spaCy model
        the strings can support multiple tokens of text but they can also use the Bert DSL.
        See the Language Options documentation: https://koaning.github.io/whatlies/tutorial/languages/#bert-style.

        Arguments:
            query: single string or list of strings

        **Usage**
        ```python
        > lang = SpacyLanguage("en_core_web_md")
        > lang['python']
        > lang[['python', 'snake']]
        > lang[['nobody expects', 'the spanish inquisition']]
        > lang = SpacyLanguage("en_trf_robertabase_lg")
        > lang['programming in [python]']
        ```
        """
        if isinstance(query, str):
            return self._get_embedding(query)
        return EmbeddingSet(*[self._get_embedding(q) for q in query])

    def _get_embedding(self, query: str) -> Embedding:
        return Embedding(query, self.model(query).vector)

    def _prepare_queries(self, prob_limit, lower):
        self._load_vocab()
        queries = [w for w in self.model.vocab]
        if prob_limit is not None:
            queries = [w for w in queries if w.prob >= prob_limit]
        if lower:
            queries = [w for w in queries if w.is_lower]
        if len(queries) == 0:
            raise ValueError(
                f"No tokens left for this setting. Consider raising prob_limit={prob_limit}"
            )
        return queries

    def _load_vocab(self):
        """This method must always be called before iterting over `model.vocab`."""
        # Load all the vocab only if they have not been loaded yet.
        if len(self.model.vocab) < len(self.model.vocab.vectors.keys()):
            for orth in self.model.vocab.vectors:
                self.model.vocab[orth]

    def _calculate_distances(self, emb, queries, metric):
        vec = emb.vector
        vector_matrix = np.array([w.vector for w in queries])
        return pairwise_distances(vector_matrix, vec.reshape(1, -1), metric=metric)

    def embset_similar(
        self,
        emb: Union[str, Embedding],
        n: int = 10,
        prob_limit=-15,
        lower=True,
        metric="cosine",
    ):
        """
        Retreive an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] that are the most simmilar to the passed query.

        Arguments:
            emb: query to use
            n: the number of items you'd like to see returned
            prob_limit: likelihood limit that sets the subset of words to search
            metric: metric to use to calculate distance, must be scipy or sklearn compatible
            lower: only fetch lower case tokens

        Returns:
            An [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] containing the similar embeddings.
        """
        embs = [w[0] for w in self.score_similar(emb, n, prob_limit, lower, metric)]
        return EmbeddingSet({w.name: w for w in embs})

    def embset_proximity(
        self,
        emb: Union[str, Embedding],
        max_proximity: float = 0.1,
        prob_limit=-15,
        lower=True,
        metric="cosine",
    ):
        """
        Retreive an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] or embeddings that are within a proximity.

        Arguments:
            emb: query to use
            max_proximity: the number of items you'd like to see returned
            prob_limit: likelihood limit that sets the subset of words to search
            metric: metric to use to calculate distance, must be scipy or sklearn compatible
            lower: only fetch lower case tokens

        Returns:
            An [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] containing the similar embeddings.
        """
        if isinstance(emb, str):
            emb = self[emb]

        queries = self._prepare_queries(prob_limit, lower)
        distances = self._calculate_distances(emb, queries, metric)
        return EmbeddingSet(
            {w: self[w] for w, d in zip(queries, distances) if d <= max_proximity}
        )

    def score_similar(
        self,
        emb: Union[str, Embedding],
        n: int = 10,
        prob_limit=-15,
        lower=True,
        metric="cosine",
    ):
        """
        Retreive a list of (Embedding, score) tuples that are the most simmilar to the passed query.

        Arguments:
            emb: query to use
            n: the number of items you'd like to see returned
            prob_limit: likelihood limit that sets the subset of words to search, to ignore set to `None`
            metric: metric to use to calculate distance, must be scipy or sklearn compatible
            lower: only fetch lower case tokens

        Returns:
            An list of ([Embedding][whatlies.embedding.Embedding], score) tuples.
        """
        if isinstance(emb, str):
            emb = self[emb]

        queries = self._prepare_queries(prob_limit, lower)
        distances = self._calculate_distances(emb, queries, metric)
        by_similarity = sorted(zip(queries, distances), key=lambda z: z[1])

        if len(queries) < n:
            warnings.warn(
                f"We could only find {len(queries)} feasible words. Consider changing `prob_limit` or `lower`",
                UserWarning,
            )

        return [(self[q.text], float(d)) for q, d in by_similarity[:n]]
