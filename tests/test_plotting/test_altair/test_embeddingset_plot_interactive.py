import json

import pytest
import numpy as np
import pandas as pd
import scipy.spatial.distance as scipy_distance

from whatlies import Embedding, EmbeddingSet
from .common import validate_plot_general_properties


"""
*Guide*

Here are the plot's propertites which could be checked (some of them may not be applicable
for a particular plot/test case):
    - type: the type of plot; usually it's scatter plot with circle marks.
    - data_field: the name of the field of chart data which is used for datapoints' coordinates.
    - data: the position (i.e. coordinates) of datapoints in the plot.
    - x_label: label of x-axis.
    - y_label: label of y-axis.
    - tilte: title of the plot.
    - label_field: the name of the field of chart data which is used for annotating data points with text labels.
    - label: the text labels used for annotation of datapoints.
    - color_field: the name of the field of chart data which is used for coloring datapoints.
"""


@pytest.fixture
def embset():
    names = ["red", "blue", "green", "yellow", "white"]
    vectors = np.random.rand(5, 4) * 10 - 5
    embeddings = [Embedding(name, vector) for name, vector in zip(names, vectors)]
    return EmbeddingSet(*embeddings)


def test_default(embset):
    p = embset.plot_interactive()
    chart = json.loads(p.to_json())
    props = {
        "type": "circle",
        "data_field": ["x_axis", "y_axis"],
        "data": embset.to_X()[:, :2],
        "x_label": "Dimension 0",
        "y_label": "Dimension 1",
        "title": "Dimension 0 vs. Dimension 1",
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "color_field": "",
    }
    chart_data = pd.DataFrame(chart["datasets"][chart["data"]["name"]])
    assert [
        chart["layer"][0]["encoding"]["x"]["field"],
        chart["layer"][0]["encoding"]["y"]["field"],
    ] == props["data_field"]
    assert np.array_equal(chart_data[["x_axis", "y_axis"]].values, props["data"])
    assert chart["layer"][0]["encoding"]["color"]["field"] == props["color_field"]
    assert chart["layer"][1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data["original"].values, props["label"])
    validate_plot_general_properties(chart["layer"][0], props)

    # Check if it's an interactive plot (done only in this test)
    assert "selection" in chart["layer"][0]
    # Check tooltip data (only done in this test case)
    tooltip_fields = set(
        [
            chart["layer"][0]["encoding"]["tooltip"][0]["field"],
            chart["layer"][0]["encoding"]["tooltip"][1]["field"],
        ]
    )
    assert tooltip_fields == set(["name", "original"])


def test_int_axis(embset):
    p = embset.plot_interactive(x_axis=2, y_axis=0, x_label="xaxis", title="some chart")
    chart = json.loads(p.to_json())
    props = {
        "type": "circle",
        "data_field": ["x_axis", "y_axis"],
        "data": np.concatenate([embset.to_X()[:, 2:3], embset.to_X()[:, :1]], axis=-1),
        "x_label": "xaxis",
        "y_label": "Dimension 0",
        "title": "some chart",
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "color_field": "",
    }
    chart_data = pd.DataFrame(chart["datasets"][chart["data"]["name"]])
    assert [
        chart["layer"][0]["encoding"]["x"]["field"],
        chart["layer"][0]["encoding"]["y"]["field"],
    ] == props["data_field"]
    assert np.array_equal(chart_data[["x_axis", "y_axis"]].values, props["data"])
    assert chart["layer"][0]["encoding"]["color"]["field"] == props["color_field"]
    assert chart["layer"][1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data["original"].values, props["label"])
    validate_plot_general_properties(chart["layer"][0], props)


def test_int_axis_with_common_str_axis_metric(embset):
    p = embset.plot_interactive(x_axis=1, y_axis=2, axis_metric="cosine_similarity")
    chart = json.loads(p.to_json())
    props = {
        "type": "circle",
        "data_field": ["x_axis", "y_axis"],
        "data": embset.to_X()[:, 1:3],
        "x_label": "Dimension 1",
        "y_label": "Dimension 2",
        "title": "Dimension 1 vs. Dimension 2",
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "color_field": "",
    }
    chart_data = pd.DataFrame(chart["datasets"][chart["data"]["name"]])
    assert [
        chart["layer"][0]["encoding"]["x"]["field"],
        chart["layer"][0]["encoding"]["y"]["field"],
    ] == props["data_field"]
    assert np.array_equal(chart_data[["x_axis", "y_axis"]].values, props["data"])
    assert chart["layer"][0]["encoding"]["color"]["field"] == props["color_field"]
    assert chart["layer"][1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data["original"].values, props["label"])
    validate_plot_general_properties(chart["layer"][0], props)


def test_str_axis(embset):
    p = embset.plot_interactive(x_axis="red", y_axis="blue")
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append([e > embset["red"], e > embset["blue"]])
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data_field": ["x_axis", "y_axis"],
        "data": vectors,
        "x_label": "red",
        "y_label": "blue",
        "title": "red vs. blue",
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "color_field": "",
    }
    chart_data = pd.DataFrame(chart["datasets"][chart["data"]["name"]])
    assert [
        chart["layer"][0]["encoding"]["x"]["field"],
        chart["layer"][0]["encoding"]["y"]["field"],
    ] == props["data_field"]
    assert np.array_equal(chart_data[["x_axis", "y_axis"]].values, props["data"])
    assert chart["layer"][0]["encoding"]["color"]["field"] == props["color_field"]
    assert chart["layer"][1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data["original"].values, props["label"])
    validate_plot_general_properties(chart["layer"][0], props)


def test_str_axis_with_common_str_axis_metric(embset):
    p = embset.plot_interactive(
        x_axis="red",
        y_axis="blue",
        y_label="blue_cosine",
        axis_metric="cosine_distance",
        color="name",
    )
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append(
            [
                scipy_distance.cosine(e.vector, embset["red"].vector),
                scipy_distance.cosine(e.vector, embset["blue"].vector),
            ]
        )
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data_field": ["x_axis", "y_axis"],
        "data": vectors,
        "x_label": "red",
        "y_label": "blue_cosine",
        "title": "red vs. blue",
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "color_field": "name",
    }
    chart_data = pd.DataFrame(chart["datasets"][chart["data"]["name"]])
    assert [
        chart["layer"][0]["encoding"]["x"]["field"],
        chart["layer"][0]["encoding"]["y"]["field"],
    ] == props["data_field"]
    assert np.array_equal(chart_data[["x_axis", "y_axis"]].values, props["data"])
    assert chart["layer"][0]["encoding"]["color"]["field"] == props["color_field"]
    assert chart["layer"][1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data["original"].values, props["label"])
    validate_plot_general_properties(chart["layer"][0], props)


def test_str_axis_with_different_axis_metric(embset):
    p = embset.plot_interactive(
        x_axis="red", y_axis="blue", axis_metric=[np.dot, "euclidean"]
    )
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append(
            [
                np.dot(e.vector, embset["red"].vector),
                scipy_distance.euclidean(e.vector, embset["blue"].vector),
            ]
        )
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data_field": ["x_axis", "y_axis"],
        "data": vectors,
        "x_label": "red",
        "y_label": "blue",
        "title": "red vs. blue",
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "color_field": "",
    }
    chart_data = pd.DataFrame(chart["datasets"][chart["data"]["name"]])
    assert [
        chart["layer"][0]["encoding"]["x"]["field"],
        chart["layer"][0]["encoding"]["y"]["field"],
    ] == props["data_field"]
    assert np.array_equal(chart_data[["x_axis", "y_axis"]].values, props["data"])
    assert chart["layer"][0]["encoding"]["color"]["field"] == props["color_field"]
    assert chart["layer"][1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data["original"].values, props["label"])
    validate_plot_general_properties(chart["layer"][0], props)


def test_emb_axis(embset):
    p = embset.plot_interactive(x_axis=embset["yellow"], y_axis=embset["white"])
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append([e > embset["yellow"], e > embset["white"]])
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data_field": ["x_axis", "y_axis"],
        "data": vectors,
        "x_label": "yellow",
        "y_label": "white",
        "title": "yellow vs. white",
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "color_field": "",
    }
    chart_data = pd.DataFrame(chart["datasets"][chart["data"]["name"]])
    assert [
        chart["layer"][0]["encoding"]["x"]["field"],
        chart["layer"][0]["encoding"]["y"]["field"],
    ] == props["data_field"]
    assert np.array_equal(chart_data[["x_axis", "y_axis"]].values, props["data"])
    assert chart["layer"][0]["encoding"]["color"]["field"] == props["color_field"]
    assert chart["layer"][1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data["original"].values, props["label"])
    validate_plot_general_properties(chart["layer"][0], props)


def test_emb_axis_with_common_str_axis_metric(embset):
    p = embset.plot_interactive(
        x_axis=embset["red"],
        y_axis=embset["green"],
        axis_metric="cosine_similarity",
        annot=False,
    )
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append(
            [
                1 - scipy_distance.cosine(e.vector, embset["red"].vector),
                1 - scipy_distance.cosine(e.vector, embset["green"].vector),
            ]
        )
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data_field": ["x_axis", "y_axis"],
        "data": vectors,
        "x_label": "red",
        "y_label": "green",
        "title": "red vs. green",
        "color_field": "",
        # Not applicable: label_field, label
    }
    chart_data = pd.DataFrame(chart["datasets"][chart["data"]["name"]])
    assert [chart["encoding"]["x"]["field"], chart["encoding"]["y"]["field"]] == props[
        "data_field"
    ]
    assert np.array_equal(chart_data[["x_axis", "y_axis"]].values, props["data"])
    assert chart["encoding"]["color"]["field"] == props["color_field"]
    assert "text" not in chart["encoding"]
    assert "layer" not in chart
    validate_plot_general_properties(chart, props)


def test_emb_axis_with_different_axis_metric(embset):
    p = embset.plot_interactive(
        x_axis=embset["blue"], y_axis=embset["yellow"], axis_metric=[None, "euclidean"]
    )
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append(
            [
                e > embset["blue"],
                scipy_distance.euclidean(e.vector, embset["yellow"].vector),
            ]
        )
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data_field": ["x_axis", "y_axis"],
        "data": vectors,
        "x_label": "blue",
        "y_label": "yellow",
        "title": "blue vs. yellow",
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "color_field": "",
    }
    chart_data = pd.DataFrame(chart["datasets"][chart["data"]["name"]])
    assert [
        chart["layer"][0]["encoding"]["x"]["field"],
        chart["layer"][0]["encoding"]["y"]["field"],
    ] == props["data_field"]
    assert np.array_equal(chart_data[["x_axis", "y_axis"]].values, props["data"])
    assert chart["layer"][0]["encoding"]["color"]["field"] == props["color_field"]
    assert chart["layer"][1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data["original"].values, props["label"])
    validate_plot_general_properties(chart["layer"][0], props)


def test_mixed_axis(embset):
    p = embset.plot_interactive(x_axis=3, y_axis="white")
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append([e.vector[3], e > embset["white"]])
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data_field": ["x_axis", "y_axis"],
        "data": vectors,
        "x_label": "Dimension 3",
        "y_label": "white",
        "title": "Dimension 3 vs. white",
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "color_field": "",
    }
    chart_data = pd.DataFrame(chart["datasets"][chart["data"]["name"]])
    assert [
        chart["layer"][0]["encoding"]["x"]["field"],
        chart["layer"][0]["encoding"]["y"]["field"],
    ] == props["data_field"]
    assert np.array_equal(chart_data[["x_axis", "y_axis"]].values, props["data"])
    assert chart["layer"][0]["encoding"]["color"]["field"] == props["color_field"]
    assert chart["layer"][1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data["original"].values, props["label"])
    validate_plot_general_properties(chart["layer"][0], props)


def test_hover_plot_basic(embset):
    """This is but a mere smoke test."""
    p = embset.plot_brush(x_axis=3, y_axis="white")
    chart = json.loads(p.to_json())
    assert "hconcat" in chart.keys()
