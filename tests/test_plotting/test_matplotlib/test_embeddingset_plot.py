import pytest
import numpy as np
import matplotlib as mpl
import scipy.spatial.distance as scipy_distance

from whatlies import Embedding, EmbeddingSet
from .common import validate_plot_general_properties


"""
*Guide*

Here are the plot's propertites which could be checked (some of them may not be applicable
for a particular plot):
    - type: the class type of collection in matplotlib to ensure the right kind of plot
        has been created.
    - data: the position of points, arrows or texts in the plot, depending on the plot's type.
    - x_label: label of x-axis.
    - y_label: label of y-axis.
    - tilte: title of the plot.
    - aspect: aspect ratio of plot, usually 'auto' unless `axis_option` argument is set.
    - color: color of points (in scatter plot) or arrows (in arrow plot). It should be rgba values.
    - label: label of points (in scatter plot) or arrows (in arrow plot).
"""


@pytest.fixture
def embset():
    names = ["red", "blue", "green", "yellow", "white"]
    vectors = np.random.rand(5, 3)
    embeddings = [Embedding(name, vector) for name, vector in zip(names, vectors)]
    return EmbeddingSet(*embeddings)


def test_embeddingset_plot_scatter_str_axis(embset):
    fig, ax = mpl.pyplot.subplots()
    embset.plot(kind="scatter", x_axis="blue", y_axis="red")
    vectors = []
    for emb in embset.embeddings.values():
        vec = []
        vec.append(emb > embset["blue"])
        vec.append(emb > embset["red"])
        vectors.append(vec)
    vectors = np.array(vectors)
    props = {
        "type": mpl.collections.PathCollection,
        "data": vectors,
        "x_label": "blue",
        "y_label": "red",
        "title": "",
        "label": list(embset.embeddings.keys()),
        "color": mpl.colors.to_rgba_array("steelblue"),
        "aspect": "auto",
    }
    assert isinstance(ax.collections[0], props["type"])
    assert np.array_equal(ax.collections[0].get_offsets(), props["data"])
    assert [t.get_text() for t in ax.texts] == props["label"]
    assert np.array_equal(ax.collections[0].get_facecolors(), props["color"])
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embeddingset_plot_arrow_str_axis(embset):
    fig, ax = mpl.pyplot.subplots()
    embset.plot(
        kind="arrow",
        x_axis="blue",
        y_axis="red",
        x_label="blue vec",
        y_label="red vec",
        title="test plot",
        color="yellow",
    )
    vectors = []
    for emb in embset.embeddings.values():
        vec = []
        vec.append(emb > embset["blue"])
        vec.append(emb > embset["red"])
        vectors.append(vec)
    vectors = np.array(vectors)
    props = {
        "type": mpl.collections.PolyCollection,
        "data": vectors,
        "x_label": "blue vec",
        "y_label": "red vec",
        "title": "test plot",
        "label": list(embset.embeddings.keys()),
        "color": mpl.colors.to_rgba_array("yellow"),
        "aspect": "auto",
    }
    UV = np.concatenate(
        (ax.collections[1].U[:, None], ax.collections[1].V[:, None]), axis=-1
    )
    assert isinstance(ax.collections[1], props["type"])
    assert np.array_equal(UV, props["data"])
    assert [t.get_text() for t in ax.texts] == props["label"]
    assert np.array_equal(ax.collections[1].get_facecolors(), props["color"])
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embeddingset_plot_scatter_integer_axis(embset):
    fig, ax = mpl.pyplot.subplots()
    embset.plot(kind="scatter", x_axis=0, y_axis=1, annot=False, color="black")
    vectors = np.concatenate((embset.to_X()[:, 0:1], embset.to_X()[:, 1:2]), axis=-1)
    props = {
        "type": mpl.collections.PathCollection,
        "data": vectors,
        "x_label": "Dimension 0",
        "y_label": "Dimension 1",
        "title": "",
        "color": mpl.colors.to_rgba_array("black"),
        "aspect": "auto",
        # Not applicable: label
    }
    assert isinstance(ax.collections[0], props["type"])
    assert np.array_equal(ax.collections[0].get_offsets(), props["data"])
    assert len(ax.texts) == 0
    assert np.array_equal(ax.collections[0].get_facecolors(), props["color"])
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embeddingset_plot_arrow_integer_axis(embset):
    fig, ax = mpl.pyplot.subplots()
    embset.plot(
        kind="arrow",
        x_axis=1,
        y_axis=2,
        x_label="1",
        y_label="2",
        color="yellow",
        axis_option="scaled",
    )
    vectors = np.concatenate((embset.to_X()[:, 1:2], embset.to_X()[:, 2:3]), axis=-1)
    props = {
        "type": mpl.collections.PolyCollection,
        "data": vectors,
        "x_label": "1",
        "y_label": "2",
        "title": "",
        "label": list(embset.embeddings.keys()),
        "color": mpl.colors.to_rgba_array("yellow"),
        "aspect": 1.0,
    }
    UV = np.concatenate(
        (ax.collections[1].U[:, None], ax.collections[1].V[:, None]), axis=-1
    )
    assert isinstance(ax.collections[1], props["type"])
    assert np.array_equal(UV, props["data"])
    assert [t.get_text() for t in ax.texts] == props["label"]
    assert np.array_equal(ax.collections[1].get_facecolors(), props["color"])
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embeddingset_plot_scatter_emb_axis(embset):
    fig, ax = mpl.pyplot.subplots()
    embset.plot(kind="scatter", x_axis=embset["green"], y_axis=embset["white"])
    vectors = []
    for emb in embset.embeddings.values():
        vec = []
        vec.append(emb > embset["green"])
        vec.append(emb > embset["white"])
        vectors.append(vec)
    vectors = np.array(vectors)
    props = {
        "type": mpl.collections.PathCollection,
        "data": vectors,
        "x_label": "green",
        "y_label": "white",
        "title": "",
        "label": list(embset.embeddings.keys()),
        "color": mpl.colors.to_rgba_array("steelblue"),
        "aspect": "auto",
    }
    assert isinstance(ax.collections[0], props["type"])
    assert np.array_equal(ax.collections[0].get_offsets(), props["data"])
    assert [t.get_text() for t in ax.texts] == props["label"]
    assert np.array_equal(ax.collections[0].get_facecolors(), props["color"])
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embeddingset_plot_arrow_emb_axis(embset):
    fig, ax = mpl.pyplot.subplots()
    embset.plot(
        kind="arrow",
        x_axis=embset["blue"],
        y_axis=embset["red"],
        x_label="xx",
        color="magenta",
    )
    vectors = []
    for emb in embset.embeddings.values():
        vec = []
        vec.append(emb > embset["blue"])
        vec.append(emb > embset["red"])
        vectors.append(vec)
    vectors = np.array(vectors)
    props = {
        "type": mpl.collections.PolyCollection,
        "data": vectors,
        "x_label": "xx",
        "y_label": "red",
        "title": "",
        "label": list(embset.embeddings.keys()),
        "color": mpl.colors.to_rgba_array("magenta"),
        "aspect": "auto",
    }
    UV = np.concatenate(
        (ax.collections[1].U[:, None], ax.collections[1].V[:, None]), axis=-1
    )
    assert isinstance(ax.collections[1], props["type"])
    assert np.array_equal(UV, props["data"])
    assert [t.get_text() for t in ax.texts] == props["label"]
    assert np.array_equal(ax.collections[1].get_facecolors(), props["color"])
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embeddingset_plot_arrow_integer_axis_with_str_axis_metric(embset):
    fig, ax = mpl.pyplot.subplots()
    embset.plot(
        kind="arrow",
        x_axis=1,
        y_axis=2,
        axis_metric="cosine_distance",
        x_label="1",
        y_label="2",
        color="yellow",
        axis_option="scaled",
    )
    vectors = np.concatenate((embset.to_X()[:, 1:2], embset.to_X()[:, 2:3]), axis=-1)
    props = {
        "type": mpl.collections.PolyCollection,
        "data": vectors,
        "x_label": "1",
        "y_label": "2",
        "title": "",
        "label": list(embset.embeddings.keys()),
        "color": mpl.colors.to_rgba_array("yellow"),
        "aspect": 1.0,
    }
    UV = np.concatenate(
        (ax.collections[1].U[:, None], ax.collections[1].V[:, None]), axis=-1
    )
    assert isinstance(ax.collections[1], props["type"])
    assert np.array_equal(UV, props["data"])
    assert [t.get_text() for t in ax.texts] == props["label"]
    assert np.array_equal(ax.collections[1].get_facecolors(), props["color"])
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embeddingset_plot_scatter_emb_axis_with_common_str_axis_metric(embset):
    fig, ax = mpl.pyplot.subplots()
    embset.plot(
        kind="scatter",
        x_axis=embset["green"],
        y_axis=embset["white"],
        axis_metric="euclidean",
    )
    vectors = []
    for emb in embset.embeddings.values():
        vectors.append(
            [
                scipy_distance.euclidean(emb.vector, embset["green"].vector),
                scipy_distance.euclidean(emb.vector, embset["white"].vector),
            ]
        )
    vectors = np.array(vectors)
    props = {
        "type": mpl.collections.PathCollection,
        "data": vectors,
        "x_label": "green",
        "y_label": "white",
        "title": "",
        "label": list(embset.embeddings.keys()),
        "color": mpl.colors.to_rgba_array("steelblue"),
        "aspect": "auto",
    }
    assert isinstance(ax.collections[0], props["type"])
    assert np.array_equal(ax.collections[0].get_offsets(), props["data"])
    assert [t.get_text() for t in ax.texts] == props["label"]
    assert np.array_equal(ax.collections[0].get_facecolors(), props["color"])
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embeddingset_plot_arrow_emb_axis_with_different_axis_metric(embset):
    fig, ax = mpl.pyplot.subplots()
    embset.plot(
        kind="arrow",
        x_axis=embset["blue"],
        y_axis="red",
        axis_metric=[scipy_distance.correlation, "cosine_similarity"],
        x_label="xx",
        color="magenta",
    )
    vectors = []
    for emb in embset.embeddings.values():
        vectors.append(
            [
                scipy_distance.correlation(emb.vector, embset["blue"].vector),
                1.0 - scipy_distance.cosine(emb.vector, embset["red"].vector),
            ]
        )
    vectors = np.array(vectors)
    props = {
        "type": mpl.collections.PolyCollection,
        "data": vectors,
        "x_label": "xx",
        "y_label": "red",
        "title": "",
        "label": list(embset.embeddings.keys()),
        "color": mpl.colors.to_rgba_array("magenta"),
        "aspect": "auto",
    }
    UV = np.concatenate(
        (ax.collections[1].U[:, None], ax.collections[1].V[:, None]), axis=-1
    )
    assert isinstance(ax.collections[1], props["type"])
    assert np.array_equal(UV, props["data"])
    assert [t.get_text() for t in ax.texts] == props["label"]
    assert np.array_equal(ax.collections[1].get_facecolors(), props["color"])
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embeddingset_plot_scatter_mixed_axis(embset):
    fig, ax = mpl.pyplot.subplots()
    embset.plot(kind="scatter", x_axis=embset["green"], y_axis=2)
    vectors = []
    for emb in embset.embeddings.values():
        vec = []
        vec.append(emb > embset["green"])
        vec.append(emb.vector[2])
        vectors.append(vec)
    vectors = np.array(vectors)
    props = {
        "type": mpl.collections.PathCollection,
        "data": vectors,
        "x_label": "green",
        "y_label": "Dimension 2",
        "title": "",
        "label": list(embset.embeddings.keys()),
        "color": mpl.colors.to_rgba_array("steelblue"),
        "aspect": "auto",
    }
    assert isinstance(ax.collections[0], props["type"])
    assert np.array_equal(ax.collections[0].get_offsets(), props["data"])
    assert [t.get_text() for t in ax.texts] == props["label"]
    assert np.array_equal(ax.collections[0].get_facecolors(), props["color"])
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embeddingset_plot_arrow_mixed_axis(embset):
    fig, ax = mpl.pyplot.subplots()
    embset.plot(kind="arrow", x_axis=1, y_axis="red", x_label="xx", color="magenta")
    vectors = []
    for emb in embset.embeddings.values():
        vec = []
        vec.append(emb.vector[1])
        vec.append(emb > embset["red"])
        vectors.append(vec)
    vectors = np.array(vectors)
    props = {
        "type": mpl.collections.PolyCollection,
        "data": vectors,
        "x_label": "xx",
        "y_label": "red",
        "title": "",
        "label": list(embset.embeddings.keys()),
        "color": mpl.colors.to_rgba_array("magenta"),
        "aspect": "auto",
    }
    UV = np.concatenate(
        (ax.collections[1].U[:, None], ax.collections[1].V[:, None]), axis=-1
    )
    assert isinstance(ax.collections[1], props["type"])
    assert np.array_equal(UV, props["data"])
    assert [t.get_text() for t in ax.texts] == props["label"]
    assert np.array_equal(ax.collections[1].get_facecolors(), props["color"])
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embeddingset_plot_raises_error_when_str_axis_not_exists(embset):
    with pytest.raises(KeyError):
        embset.plot(x_axis="bnb", y_axis="blue")
    with pytest.raises(KeyError):
        embset.plot(x_axis="red", y_axis="clk")
