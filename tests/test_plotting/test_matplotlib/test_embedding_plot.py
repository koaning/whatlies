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


def test_embedding_plot_scatter_integer_axis(embset):
    emb = embset["red"]
    fig, ax = mpl.pyplot.subplots()
    emb.plot(kind="scatter", x_axis=0, y_axis=1)
    props = {
        "type": mpl.collections.PathCollection,
        "data": emb.vector[0:2],
        "x_label": "Dimension 0",
        "y_label": "Dimension 1",
        "title": "",
        "color": mpl.colors.to_rgba_array("steelblue"),
        "label": "red",
        "aspect": "auto",
    }
    assert np.array_equal(ax.collections[0].get_offsets()[0], props["data"])
    assert isinstance(ax.collections[0], props["type"])
    assert np.array_equal(ax.collections[0].get_facecolor(), props["color"])
    assert ax.texts[0].get_text() == props["label"]
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embedding_plot_arrow_integer_axis(embset):
    emb = embset["red"]
    fig, ax = mpl.pyplot.subplots()
    emb.plot(
        kind="arrow",
        x_axis=0,
        y_axis=2,
        color="blue",
        x_label="xlabel",
        y_label="ylabel",
        title="test plot",
        annot=False,
    )
    props = {
        "type": mpl.collections.PolyCollection,
        "data": np.concatenate((emb.vector[0:1], emb.vector[2:3])),
        "x_label": "xlabel",
        "y_label": "ylabel",
        "title": "test plot",
        "color": mpl.colors.to_rgba_array("blue"),
        "aspect": "auto",
        # Not applicable: label
    }
    UV = np.concatenate((ax.collections[1].U, ax.collections[1].V))
    assert isinstance(ax.collections[1], props["type"])
    assert np.array_equal(UV, props["data"])
    assert np.array_equal(ax.collections[1].get_facecolor(), props["color"])
    assert len(ax.texts) == 0
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embedding_plot_text_integer_axis(embset):
    emb = embset["red"]
    fig, ax = mpl.pyplot.subplots()
    emb.plot(kind="text", x_axis=1, y_axis=2)
    props = {
        "data": np.concatenate((emb.vector[1:2] + 0.01, emb.vector[2:3])),
        "x_label": "Dimension 1",
        "y_label": "Dimension 2",
        "title": "",
        "label": "red",
        "aspect": "auto",
        # Not applicable: type, color
    }
    assert np.array_equal(ax.texts[0].get_position(), props["data"])
    assert len(ax.collections) == 0
    assert ax.texts[0].get_text() == props["label"]
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embedding_plot_scatter_emb_axis(embset):
    emb = embset["red"]
    fig, ax = mpl.pyplot.subplots()
    emb.plot(kind="scatter", x_axis=embset["blue"], y_axis=embset["green"])
    props = {
        "type": mpl.collections.PathCollection,
        "data": np.array([emb > embset["blue"], emb > embset["green"]]),
        "x_label": "blue",
        "y_label": "green",
        "color": mpl.colors.to_rgba_array("steelblue"),
        "title": "",
        "label": "red",
        "aspect": "auto",
    }
    assert np.array_equal(ax.collections[0].get_offsets()[0], props["data"])
    assert isinstance(ax.collections[0], props["type"])
    assert ax.texts[0].get_text() == props["label"]
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embedding_plot_arrow_emb_axis(embset):
    emb = embset["red"] + embset["yellow"]
    fig, ax = mpl.pyplot.subplots()
    emb.plot(
        kind="arrow",
        x_axis=embset["blue"],
        y_axis=embset["green"],
        color="yellow",
        show_ops=True,
        axis_option="equal",
    )
    props = {
        "type": mpl.collections.PolyCollection,
        "data": np.array([emb > embset["blue"], emb > embset["green"]]),
        "x_label": "blue",
        "y_label": "green",
        "color": mpl.colors.to_rgba_array("yellow"),
        "title": "",
        "label": "(red + yellow)",
        "aspect": 1.0,
    }
    UV = np.concatenate((ax.collections[1].U, ax.collections[1].V))
    assert isinstance(ax.collections[1], props["type"])
    assert np.array_equal(UV, props["data"])
    assert np.array_equal(ax.collections[1].get_facecolor(), props["color"])
    assert ax.texts[0].get_text() == props["label"]
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embedding_plot_arrow_integer_axis_with_str_axis_metric(embset):
    emb = embset["red"]
    fig, ax = mpl.pyplot.subplots()
    emb.plot(
        kind="arrow",
        x_axis=0,
        y_axis=2,
        axis_metric="euclidean",
        color="blue",
        x_label="xlabel",
        y_label="ylabel",
        title="test plot",
        annot=False,
    )
    props = {
        "type": mpl.collections.PolyCollection,
        "data": np.concatenate((emb.vector[0:1], emb.vector[2:3])),
        "x_label": "xlabel",
        "y_label": "ylabel",
        "title": "test plot",
        "color": mpl.colors.to_rgba_array("blue"),
        "aspect": "auto",
        # Not applicable: label
    }
    UV = np.concatenate((ax.collections[1].U, ax.collections[1].V))
    assert isinstance(ax.collections[1], props["type"])
    assert np.array_equal(UV, props["data"])
    assert np.array_equal(ax.collections[1].get_facecolor(), props["color"])
    assert len(ax.texts) == 0
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embedding_plot_scatter_emb_axis_with_common_str_axis_metric(embset):
    emb = embset["red"]
    fig, ax = mpl.pyplot.subplots()
    emb.plot(
        kind="scatter",
        x_axis=embset["blue"],
        y_axis=embset["green"],
        axis_metric="cosine_distance",
    )
    props = {
        "type": mpl.collections.PathCollection,
        "data": np.array(
            [
                scipy_distance.cosine(emb.vector, embset["blue"].vector),
                scipy_distance.cosine(emb.vector, embset["green"].vector),
            ]
        ),
        "x_label": "blue",
        "y_label": "green",
        "color": mpl.colors.to_rgba_array("steelblue"),
        "title": "",
        "label": "red",
        "aspect": "auto",
    }
    assert np.array_equal(ax.collections[0].get_offsets()[0], props["data"])
    assert isinstance(ax.collections[0], props["type"])
    assert ax.texts[0].get_text() == props["label"]
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embedding_plot_arrow_emb_axis_with_different_str_axis_metric(embset):
    emb = embset["red"] + embset["yellow"]
    fig, ax = mpl.pyplot.subplots()
    emb.plot(
        kind="arrow",
        x_axis=embset["blue"],
        y_axis=embset["green"],
        axis_metric=["euclidean", "cosine_similarity"],
        color="yellow",
        show_ops=True,
        axis_option="equal",
    )
    props = {
        "type": mpl.collections.PolyCollection,
        "data": np.array(
            [
                scipy_distance.euclidean(emb.vector, embset["blue"].vector),
                1.0 - scipy_distance.cosine(emb.vector, embset["green"].vector),
            ]
        ),
        "x_label": "blue",
        "y_label": "green",
        "color": mpl.colors.to_rgba_array("yellow"),
        "title": "",
        "label": "(red + yellow)",
        "aspect": 1.0,
    }
    UV = np.concatenate((ax.collections[1].U, ax.collections[1].V))
    assert isinstance(ax.collections[1], props["type"])
    assert np.array_equal(UV, props["data"])
    assert np.array_equal(ax.collections[1].get_facecolor(), props["color"])
    assert ax.texts[0].get_text() == props["label"]
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embedding_plot_arrow_emb_axis_with_callable_y_axis_metric(embset):
    emb = embset["red"] + embset["yellow"]
    fig, ax = mpl.pyplot.subplots()
    emb.plot(
        kind="arrow",
        x_axis=embset["blue"],
        y_axis=embset["green"],
        axis_metric=[None, scipy_distance.correlation],
        x_label="xaxis",
        y_label="corr",
        color="yellow",
        show_ops=True,
        axis_option="equal",
    )
    props = {
        "type": mpl.collections.PolyCollection,
        "data": np.array(
            [
                emb > embset["blue"],
                scipy_distance.correlation(emb.vector, embset["green"].vector),
            ]
        ),
        "x_label": "xaxis",
        "y_label": "corr",
        "color": mpl.colors.to_rgba_array("yellow"),
        "title": "",
        "label": "(red + yellow)",
        "aspect": 1.0,
    }
    UV = np.concatenate((ax.collections[1].U, ax.collections[1].V))
    assert isinstance(ax.collections[1], props["type"])
    assert np.array_equal(UV, props["data"])
    assert np.array_equal(ax.collections[1].get_facecolor(), props["color"])
    assert ax.texts[0].get_text() == props["label"]
    validate_plot_general_properties(ax, props)
    mpl.pyplot.close(fig)


def test_embedding_plot_raises_error_when_incorrect_axis_type(embset):
    emb = embset["red"]
    with pytest.raises(ValueError, match="The `x_axis` value should be"):
        emb.plot(x_axis=1.0)
    with pytest.raises(ValueError, match="The `y_axis` value should be"):
        emb.plot(y_axis="blue")


def test_embedding_plot_raises_error_when_incorrect_axis_metric(embset):
    emb = embset["red"]
    with pytest.raises(ValueError, match="The given axis metric is not supported"):
        emb.plot(x_axis=embset["blue"], axis_metric="correlation")
    with pytest.raises(ValueError, match="The given axis metric type is not"):
        emb.plot(y_axis=embset["blue"], axis_metric=1)
