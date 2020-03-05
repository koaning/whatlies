import spacy
from whatlies.embedding import Embedding


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
        start, end = 0, -1
        split_string = string.split(" ")
        for idx, word in enumerate(split_string):
            if word[0] == "[":
                start = idx
            if word[-1] == "]":
                end = idx + 1
        return Embedding(string, doc[start:end].vector)
