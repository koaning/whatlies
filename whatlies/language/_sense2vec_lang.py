import spacy
from sense2vec import Sense2Vec, Sense2VecComponent

from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet


class Sense2VecLanguage:
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

    Warning:
        This tool is temporarily not supported because sense2vec isn't supported by spaCy v3 just yet.

    """

    def __init__(self, sense2vec_path):
        self.sense2vec_path = sense2vec_path
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
