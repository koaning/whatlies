import pathlib

import pytest
import fasttext

from whatlies.language import FasttextLanguage

text_path = str(pathlib.Path(__file__).parent.parent.absolute() / "data" / "foobar.txt")
model1 = fasttext.train_unsupervised(
    text_path, model="cbow", dim=20, epoch=20, min_count=1
)
model2 = fasttext.train_unsupervised(text_path, model="skipgram", dim=10, min_count=1)


def test_load_in_model1():
    lang = FasttextLanguage(model1)
    assert lang["dog"].vector.shape[0] == 20


def test_load_in_model2():
    lang = FasttextLanguage(model2)
    assert lang["dog"].vector.shape[0] == 10


def test_retreive_similar_len():
    assert len(FasttextLanguage(model1).score_similar("cat", 20)) == 20
    assert len(FasttextLanguage(model2).score_similar("cat", 10)) == 10
    assert len(FasttextLanguage(model1).score_similar("cat", 1000)) == 91
    assert len(FasttextLanguage(model2).score_similar("cat", 1000)) == 91


def test_raise_warning():
    with pytest.warns(UserWarning):
        FasttextLanguage(model1).score_similar("cat", 1000)
