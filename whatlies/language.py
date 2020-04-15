from typing import Union

import spacy
import numpy as np
from typing import Union, List
from sklearn.metrics import pairwise_distances
from sense2vec import Sense2Vec, Sense2VecComponent

from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet


def _selected_idx_spacy(string):
    if "[" not in string:
        if "]" not in string:
            return 0, len(string.split(" "))
    start, end = 0, -1
    split_string = string.split(" ")
    for idx, word in enumerate(split_string):
        if word[0] == "[":
            start = idx
        if word[-1] == "]":
            end = idx + 1
    return start, end


class SpacyLanguage:
    """
    This object is used to lazily fetch [Embedding][whatlies.embedding.Embedding]s or
    [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]s from a spaCy language
    backend. This object is meant for retreival, not plotting.

    Arguments:
        model_name: name of the model to load, be sure that it's downloaded beforehand

    **Usage**:

    ```python
    > lang = SpacyLanguage("en_core_web_md")
    > lang['python']
    > lang[['python', 'snake', 'dog']]

    > lang = SpacyLanguage("en_trf_robertabase_lg")
    > lang['programming in [python]']
    ```
    """

    def __init__(self, model_name):
        self.nlp = spacy.load(model_name)

    @staticmethod
    def _input_str_legal(string):
        if sum(1 for c in string if c == "[") > 1:
            raise ValueError("only one opener `[` allowed ")
        if sum(1 for c in string if c == "]") > 1:
            raise ValueError("only one opener `]` allowed ")

    def __getitem__(self, query: Union[str, List[str]]):
        """
        Retreive a single embedding or a set of embeddings. Depending on the spaCy model
        the strings can support multiple tokens of text but they can also use the Bert DSL.
        See the Language Options documentation: https://rasahq.github.io/whatlies/tutorial/languages/#bert-style.

        Arguments:
            query: single string or list of strings

        **Usage**
        ```python
        > lang = SpacyLanguage("en_core_web_md")
        > lang['python']
        > lang[['python'], ['snake']]
        > lang[['nobody expects'], ['the spanish inquisition']]
        ```
        """
        if isinstance(query, str):
            self._input_str_legal(query)
            start, end = _selected_idx_spacy(query)
            clean_string = query.replace("[", "").replace("]", "")
            vec = self.nlp(clean_string)[start:end].vector
            return Embedding(query, vec)
        return EmbeddingSet(*[self[tok] for tok in query])

    def embset_similar(self, emb: Union[str, Embedding], n: int = 10, prob_limit=-15, lower=True, metric='cosine'):
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

    def score_similar(self, emb: Union[str, Embedding], n: int = 10, prob_limit=-15, lower=True, metric='cosine'):
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

        vec = emb.vector
        queries = [w for w in self.nlp.vocab]
        if prob_limit is not None:
            queries = [w for w in queries if w.prob >= prob_limit]
        if lower:
            queries = [w for w in queries if w.is_lower]
        if len(queries) == 0:
            raise ValueError(f"Language model has no tokens for this setting. Consider raising prob_limit={prob_limit}")

        vector_matrix = np.array([w.vector for w in queries])
        distances = pairwise_distances(vector_matrix, vec.reshape(1, -1), metric=metric)
        by_similarity = sorted(zip(queries, distances), key=lambda z: z[1])

        return [(self[q.text], float(d)) for q, d in by_similarity[:n]]


class Sense2VecLangauge:
    """
    This object is used to lazily fetch [Embedding][whatlies.embedding.Embedding]s or
    [EmbeddingSet][whatlies.embeddingset.EmbeddingSet]s from a sense2vec language
    backend. This object is meant for retreival, not plotting.

    Arguments:
        sense2vec_path: path to downloaded vectors

    **Usage**:

    ```python
    > lang = Sense2VecLanguage(sense2vec_path="/path/to/reddit_vectors-1.1.0")
    > lang['bank|NOUN']
    > lang['bank|VERB']
    ```

    Important:
        The reddit vectors are not given by this library.
        You can find the download link [here](https://github.com/explosion/sense2vec#pretrained-vectors).

    """

    def __init__(self, sense2vec_path):
        self.s2v = Sense2Vec().from_disk(sense2vec_path)

    def __getitem__(self, query):
        """
        Retreive a single embedding or a set of embeddings.

        Arguments:
            query: single string or list of strings

        **Usage**
        ```python
        > lang = SpacyLanguage("en_core_web_md")
        > lang['duck|NOUN']
        > lang[['duck|NOUN'], ['duck|VERB']]
        ```
        """
        if isinstance(query, str):
            vec = self.s2v[query]
            return Embedding(query, vec)
        return EmbeddingSet(*[self[tok] for tok in query])

    def embset_similar(self, query, n=10):
        """
        Retreive an [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] that are the most simmilar to the passed query.

        Arguments:
            query: query to use
            n: the number of items you'd like to see returned

        Returns:
            An [EmbeddingSet][whatlies.embeddingset.EmbeddingSet] containing the similar embeddings.
        """
        return EmbeddingSet(
            *[self[tok] for tok, sim in self.s2v.most_similar(query, n=n)],
            name=f"Embset[s2v similar_{n}:{query}]",
        )

    def score_similar(self, query, n=10):
        """
        Retreive an EmbeddingSet that are the most simmilar to the passed query.

        Arguments:
            query: query to use
            n: the number of items you'd like to see returned

        Returns:
            An list of ([Embedding][whatlies.embedding.Embedding], score) tuples.
        """
        return [(self[tok], sim) for tok, sim in self.s2v.most_similar(query, n=n)]


class Sense2VecSpacyLanguage:
    """
    This object is used to lazily fetch `Embedding`s from a sense2vec language
    backend. Note that it is different than an `EmbeddingSet` in the sense
    it does not have anything precomputed.
    **Usage**:
    ```
    lang = Sense2VecLanguage(spacy_model="en_core_web_sm", sense2vec="/path/to/reddit_vectors-1.1.0")
    lang['bank|NOUN']
    lang['bank|VERB']
    ```
    """

    def __init__(self, model_name, sense2vec_path):
        self.nlp = spacy.load(model_name)
        s2v = Sense2VecComponent(self.nlp.vocab).from_disk(sense2vec_path)
        self.nlp.add(s2v)

    def __getitem__(self, string):
        doc = self.nlp(string)
        vec = doc.vector
        start, end = 0, -1
        split_string = string.split(" ")
        for idx, word in enumerate(split_string):
            if word[0] == "[":
                start = idx
            if word[-1] == "]":
                end = idx + 1
        if start != 0:
            if end != -1:
                vec = doc[start:end].vector
        return Embedding(string, vec)
