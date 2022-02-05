import pytest
import numpy as np

from whatlies import Embedding, EmbeddingSet


@pytest.fixture
def emb():
    x = Embedding("x", [0.0, 1.0])
    y = Embedding("y", [1.0, 0.0])
    z = Embedding("z", [0.5, 0.5])
    return EmbeddingSet(x, y, z)


def test_emb_dist(emb):
    assert np.isclose(emb["x"].distance(emb["x"]), 0.0)
    assert np.isclose(emb["x"].distance(emb["y"], metric="euclidean"), np.sqrt(2))
    assert np.isclose(emb["x"].distance(emb["z"], metric="euclidean"), np.sqrt(2) / 2)


def test_emb_norm(emb):
    assert np.isclose(emb["x"].norm, 1.0)
    assert np.isclose(emb["y"].norm, 1.0)
    assert np.isclose(emb["z"].norm, np.sqrt(0.5))


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


def test_emb_gt(emb):
    assert (emb["z"] > emb["x"]) == pytest.approx(0.5)
    assert (emb["x"] > emb["z"]) == pytest.approx(1.0)


def test_emb_plot_no_err_2d(emb):
    emb["x"].plot(kind="arrow").plot(kind="text")
    emb["y"].plot(kind="arrow").plot(kind="text")
    emb["z"].plot(kind="arrow").plot(kind="text")


def test_emb_plot_no_err_3d():
    x = Embedding("x", [0.0, 1.0, 1.0])
    y = Embedding("y", [1.0, 0.0, 1.0])
    z = Embedding("z", [0.5, 0.5, 1.0])
    for item in [x, y, z]:
        item.plot("scatter", x_axis=x, y_axis=y)


def test_emb_str_method(emb):
    for char in "xyz":
        assert str(emb[char]) == char


def test_emb_ndim():
    foo = Embedding("foo", [0, 1, 0.2])
    assert foo.ndim == 3


def test_negation():
    foo = Embedding("foo", [0.1, 0.3])
    assert np.allclose((-foo).vector, -np.array([0.1, 0.3]))
