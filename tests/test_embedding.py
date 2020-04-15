import pytest
import numpy as np

from whatlies import Embedding, EmbeddingSet


@pytest.fixture
def emb():
    x = Embedding("x", [0.0, 1.0])
    y = Embedding("y", [1.0, 0.0])
    z = Embedding("z", [0.5, 0.5])
    return EmbeddingSet(x, y, z)


def test_emb_add(emb):
    new_emb = emb["x"] + emb["y"]
    assert np.isclose(new_emb.vector, np.array([1.0, 1.0])).all()


def test_emb_subtract(emb):
    new_emb = emb["x"] - emb["y"]
    assert np.isclose(new_emb.vector, np.array([-1.0, 1.0])).all()


def test_emb_proj_unto1(emb):
    new_emb = emb["z"] >> emb["y"]
    assert np.isclose(new_emb.vector, np.array([0.5, 0.0])).all()


def test_emb_proj_unto2(emb):
    new_emb = emb["z"] >> emb["x"]
    assert np.isclose(new_emb.vector, np.array([0.0, 0.5])).all()


def test_emb_proj_away1(emb):
    new_emb = emb["z"] | emb["y"]
    assert np.isclose(new_emb.vector, np.array([0.0, 0.5])).all()


def test_emb_proj_away2(emb):
    new_emb = emb["z"] | emb["x"]
    assert np.isclose(new_emb.vector, np.array([0.5, 0.0])).all()
