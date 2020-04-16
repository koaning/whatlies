"""
This script adds a small spaCy model that is used in testing.
"""

import spacy
from spacy.language import Language
from spacy.vocab import Vocab

if __name__ == "__main__":
    words = [
        "prince",
        "princess",
        "nurse",
        "doctor",
        "banker",
        "man",
        "woman",
        "cousin",
        "neice",
        "king",
        "queen",
        "dude",
        "guy",
        "gal",
        "fire",
        "dog",
        "cat",
        "mouse",
        "red",
        "bluee",
        "green",
        "yellow",
        "water",
        "person",
        "family",
        "brother",
        "sister",
    ]
    nlp = spacy.load("en_core_web_md")
    vec_data = {w: nlp(w).vector for w in words}
    vocab = Vocab(strings=words)
    for word, vector in vec_data.items():
        vocab.set_vector(word, vector)
    nlp = Language(vocab=vocab, meta={"lang": "en"})
    vocab.to_disk("custom_test_vocab")
