import pytest
import numpy as np

from spacy.vocab import Vocab
from spacy.language import Language
from whatlies.language import SpacyLanguage


@pytest.fixture(scope="module")
def lang():
    return SpacyLanguage("en_core_web_md")


def test_basic_usage(lang):
    queries = [
        "yesterday is history",
        "tomorrow is mystery",
        "today is a gift",
        "that's why it's called [present]",
    ]
    emb = lang[queries]
    assert len(emb) == 4
    assert emb[queries[0]].name == "yesterday is history"
    assert emb[queries[0]].vector.shape == (300,)


@pytest.mark.parametrize(
    "query",
    ["no [closing bracket", "no opening] bracket", "[more] than one [context]",],
)
def test_invalid_query_raises_error(lang, query):
    with pytest.raises(ValueError, match="bracket"):
        lang[query]


@pytest.mark.parametrize(
    "query, context, context_pos",
    [
        ("I'm going [home]", "home", (3, 4)),
        ("you should pre-order your [washing machine]", "washing machine", (6, 8)),
        ("today is the [best day] of my life", "best day", (3, 5)),
        ("[yesterday] isn't mystery", "yesterday", (0, 1)),
        (
            "Let's have Fun without context.",
            "Let's have Fun without context.",
            (0, None),
        ),
    ],
)
def test_embedding_is_correct(lang, query, context, context_pos):
    emb = lang[query]
    assert emb.vector.shape == (300,)
    assert emb.name == query

    clean_query = query.replace("[", "").replace("]", "")
    doc = lang.model(clean_query)
    span = doc[context_pos[0] : context_pos[1]]
    assert str(span) == context
    assert np.allclose(emb.vector, span.vector)


@pytest.fixture()
def color_lang():
    vector_data = {
        "red": np.array([1.0, 0.0]),
        "green": np.array([0.5, 0.5]),
        "blue": np.array([0.0, 1.0]),
        "purple": np.array([0.0, 1.0]),
    }

    vocab = Vocab(strings=vector_data.keys())
    for word, vector in vector_data.items():
        vocab.set_vector(word, vector)
    nlp = Language(vocab=vocab)
    return SpacyLanguage(nlp)


def test_score_similar_one(color_lang):
    scores = color_lang.score_similar("blue", n=2, prob_limit=None, lower=False)
    print(scores)
    assert all([s[1] == 0 for s in scores])
    assert "blue" in [s[0].name for s in scores]
    assert "purple" in [s[0].name for s in scores]


@pytest.mark.parametrize(
    "string, array",
    zip(
        ["red", "red green [blue] purple", "green [red blue] pink"],
        [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5, 0.5])],
    ),
)
def test_lang_retreival(color_lang, string, array):
    assert np.isclose(color_lang[string].vector, array).all()


def test_single_token_words(color_lang):
    # test for issue here: https://github.com/RasaHQ/whatlies/issues/5
    assert len(color_lang["red"].vector) > 0


def test_raise_warning(color_lang):
    print([w for w in color_lang.nlp.vocab])
    with pytest.warns(UserWarning):
        color_lang.score_similar("red", 100, prob_limit=None, lower=False)
