import spacy
from sense2vec import Sense2Vec, Sense2VecComponent
from whatlies.embedding import Embedding
from whatlies.embeddingset import EmbeddingSet


class SpacyLanguage:
    """
    This object is used to lazily fetch `Embedding`s from a spaCy language
    backend. Note that it is different than an `EmbeddingSet` in the sense
    it does not have anything precomputed.
    **Usage**:
    ```
    lang = SpacyLanguage("en_core_web_md")
    lang['python']
    lang = SpacyLanguage("en_trf_robertabase_lg")
    lang['programming in [python]']
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

    def __getitem__(self, string):
        self._input_str_legal(string)
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


class Sense2VecLangauge:
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
    def __init__(self, sense2vec_path):
        self.s2v = Sense2Vec().from_disk(sense2vec_path)

    def __getitem__(self, string):
        vec = self.s2v[string]
        return Embedding(string, vec)

    def embset_similar(self, query, n=10):
        return EmbeddingSet(*[self[tok] for tok, sim in self.s2v.most_similar(query, n=n)],
                            name=f"Embset[s2v similar_{n}:{query}]")

    def score_similar(self, query, n=10):
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
