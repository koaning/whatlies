import json

import pytest
import numpy as np
import pandas as pd
import scipy.spatial.distance as scipy_distance

from whatlies import Embedding, EmbeddingSet
from .common import validate_matrix_plot_general_properties


"""
*Guide*

Here are the plot's propertites which could be checked (some of them may not be applicable
for a particular plot/test case):
    - type: the type of plot; usually it's scatter plot with circle marks.
    - data: the position (i.e. coordinates) of datapoints in the plot.
    - x_repeat: the name of repeat field representing x-axis.
    - x_repeat_fields: the name of the fields of chart data which is used for x-axis coordinates.
    - y_repeat: the name of repeat field representing y-axis.
    - y_repeat_fields: the name of the fields of chart data which is used for y-axis coordinates.
    - label_field: the name of the field of chart data which is used for annotating data points with text labels.
    - label: the text labels used for annotation of datapoints.
    - width: the width of plot
    - height: the height of plot
"""


@pytest.fixture
def embset():
    names = ["red", "blue", "green", "yellow", "white"]
    vectors = np.random.rand(5, 4) * 10 - 5
    embeddings = [Embedding(name, vector) for name, vector in zip(names, vectors)]
    return EmbeddingSet(*embeddings)


def test_default(embset):
    p = embset.plot_interactive_matrix()
    chart = json.loads(p.to_json())
    props = {
        "type": "circle",
        "data": embset.to_X()[:, :2],
        "x_repeat": "column",
        "x_repeat_fields": ["Dimension 0", "Dimension 1"],
        "y_repeat": "row",
        "y_repeat_fields": ["Dimension 1", "Dimension 0"],
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "height": 200,
        "width": 200,
    }
    layers = chart["spec"]["layer"]
    chart_data = pd.DataFrame(chart["datasets"][chart["spec"]["data"]["name"]])
    assert layers[0]["mark"] == props["type"]
    assert np.array_equal(chart_data[props["x_repeat_fields"]].values, props["data"])
    assert layers[0]["encoding"]["x"]["field"]["repeat"] == props["x_repeat"]
    assert layers[0]["encoding"]["y"]["field"]["repeat"] == props["y_repeat"]
    assert chart["repeat"]["column"] == props["x_repeat_fields"]
    assert chart["repeat"]["row"] == props["y_repeat_fields"]
    assert layers[1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data[props["label_field"]], props["label"])
    validate_matrix_plot_general_properties(chart["spec"], props)

    # Check if it's an interactive plot (only done in this test case)
    assert "selection" in layers[0]
    # Check tooltip data (only done in this test case)
    tooltip_fields = set(
        [
            layers[0]["encoding"]["tooltip"][0]["field"],
            layers[0]["encoding"]["tooltip"][1]["field"],
        ]
    )
    assert tooltip_fields == set(["name", "original"])


def test_int_axis(embset):
    p = embset.plot_interactive_matrix(2, 3, 0, height=300)
    chart = json.loads(p.to_json())
    props = {
        "type": "circle",
        "data": embset.to_X()[:, [2, 3, 0]],
        "x_repeat": "column",
        "x_repeat_fields": ["Dimension 2", "Dimension 3", "Dimension 0"],
        "y_repeat": "row",
        "y_repeat_fields": ["Dimension 0", "Dimension 3", "Dimension 2"],
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "height": 300,
        "width": 200,
    }
    layers = chart["spec"]["layer"]
    chart_data = pd.DataFrame(chart["datasets"][chart["spec"]["data"]["name"]])
    assert layers[0]["mark"] == props["type"]
    assert np.array_equal(chart_data[props["x_repeat_fields"]].values, props["data"])
    assert layers[0]["encoding"]["x"]["field"]["repeat"] == props["x_repeat"]
    assert layers[0]["encoding"]["y"]["field"]["repeat"] == props["y_repeat"]
    assert chart["repeat"]["column"] == props["x_repeat_fields"]
    assert chart["repeat"]["row"] == props["y_repeat_fields"]
    assert layers[1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data[props["label_field"]], props["label"])
    validate_matrix_plot_general_properties(chart["spec"], props)


def test_int_axis_with_str_axes_metric(embset):
    p = embset.plot_interactive_matrix(3, 2, 1, axes_metric="euclidean", annot=False)
    chart = json.loads(p.to_json())
    props = {
        "type": "circle",
        "data": embset.to_X()[:, 3:0:-1],
        "x_repeat": "column",
        "x_repeat_fields": ["Dimension 3", "Dimension 2", "Dimension 1"],
        "y_repeat": "row",
        "y_repeat_fields": ["Dimension 1", "Dimension 2", "Dimension 3"],
        "height": 200,
        "width": 200,
        # Not applicabel: label_field, label
    }
    chart_data = pd.DataFrame(chart["datasets"][chart["spec"]["data"]["name"]])
    assert chart["spec"]["mark"] == props["type"]
    assert np.array_equal(chart_data[props["x_repeat_fields"]].values, props["data"])
    assert chart["spec"]["encoding"]["x"]["field"]["repeat"] == props["x_repeat"]
    assert chart["spec"]["encoding"]["y"]["field"]["repeat"] == props["y_repeat"]
    assert chart["repeat"]["column"] == props["x_repeat_fields"]
    assert chart["repeat"]["row"] == props["y_repeat_fields"]
    assert "text" not in chart["spec"]["encoding"]
    assert "layer" not in chart["spec"]
    validate_matrix_plot_general_properties(chart["spec"], props)


def test_str_axis(embset):
    p = embset.plot_interactive_matrix("red", "blue", "green")
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append([e > embset["red"], e > embset["blue"], e > embset["green"]])
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data": vectors,
        "x_repeat": "column",
        "x_repeat_fields": ["red", "blue", "green"],
        "y_repeat": "row",
        "y_repeat_fields": ["green", "blue", "red"],
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "height": 200,
        "width": 200,
    }
    layers = chart["spec"]["layer"]
    chart_data = pd.DataFrame(chart["datasets"][chart["spec"]["data"]["name"]])
    assert layers[0]["mark"] == props["type"]
    assert np.array_equal(chart_data[props["x_repeat_fields"]].values, props["data"])
    assert layers[0]["encoding"]["x"]["field"]["repeat"] == props["x_repeat"]
    assert layers[0]["encoding"]["y"]["field"]["repeat"] == props["y_repeat"]
    assert chart["repeat"]["column"] == props["x_repeat_fields"]
    assert chart["repeat"]["row"] == props["y_repeat_fields"]
    assert layers[1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data[props["label_field"]], props["label"])
    validate_matrix_plot_general_properties(chart["spec"], props)


def test_str_axis_with_str_axes_metric(embset):
    p = embset.plot_interactive_matrix(
        "yellow", "blue", "green", axes_metric="cosine_similarity", width=350
    )
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append(
            [
                1 - scipy_distance.cosine(e.vector, embset["yellow"].vector),
                1 - scipy_distance.cosine(e.vector, embset["blue"].vector),
                1 - scipy_distance.cosine(e.vector, embset["green"].vector),
            ]
        )
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data": vectors,
        "x_repeat": "column",
        "x_repeat_fields": ["yellow", "blue", "green"],
        "y_repeat": "row",
        "y_repeat_fields": ["green", "blue", "yellow"],
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "height": 200,
        "width": 350,
    }
    layers = chart["spec"]["layer"]
    chart_data = pd.DataFrame(chart["datasets"][chart["spec"]["data"]["name"]])
    assert layers[0]["mark"] == props["type"]
    assert np.array_equal(chart_data[props["x_repeat_fields"]].values, props["data"])
    assert layers[0]["encoding"]["x"]["field"]["repeat"] == props["x_repeat"]
    assert layers[0]["encoding"]["y"]["field"]["repeat"] == props["y_repeat"]
    assert chart["repeat"]["column"] == props["x_repeat_fields"]
    assert chart["repeat"]["row"] == props["y_repeat_fields"]
    assert layers[1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data[props["label_field"]], props["label"])
    validate_matrix_plot_general_properties(chart["spec"], props)


def test_str_axis_with_different_axes_metric(embset):
    p = embset.plot_interactive_matrix(
        "yellow", "blue", "white", axes_metric=["cosine_distance", None, np.dot]
    )
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append(
            [
                scipy_distance.cosine(e.vector, embset["yellow"].vector),
                e > embset["blue"],
                np.dot(e.vector, embset["white"].vector),
            ]
        )
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data": vectors,
        "x_repeat": "column",
        "x_repeat_fields": ["yellow", "blue", "white"],
        "y_repeat": "row",
        "y_repeat_fields": ["white", "blue", "yellow"],
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "height": 200,
        "width": 200,
    }
    layers = chart["spec"]["layer"]
    chart_data = pd.DataFrame(chart["datasets"][chart["spec"]["data"]["name"]])
    assert layers[0]["mark"] == props["type"]
    assert np.array_equal(chart_data[props["x_repeat_fields"]].values, props["data"])
    assert layers[0]["encoding"]["x"]["field"]["repeat"] == props["x_repeat"]
    assert layers[0]["encoding"]["y"]["field"]["repeat"] == props["y_repeat"]
    assert chart["repeat"]["column"] == props["x_repeat_fields"]
    assert chart["repeat"]["row"] == props["y_repeat_fields"]
    assert layers[1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data[props["label_field"]], props["label"])
    validate_matrix_plot_general_properties(chart["spec"], props)


def test_emb_axis(embset):
    p = embset.plot_interactive_matrix(embset["blue"], embset["green"])
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append([e > embset["blue"], e > embset["green"]])
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data": vectors,
        "x_repeat": "column",
        "x_repeat_fields": ["blue", "green"],
        "y_repeat": "row",
        "y_repeat_fields": ["green", "blue"],
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "height": 200,
        "width": 200,
    }
    layers = chart["spec"]["layer"]
    chart_data = pd.DataFrame(chart["datasets"][chart["spec"]["data"]["name"]])
    assert layers[0]["mark"] == props["type"]
    assert np.array_equal(chart_data[props["x_repeat_fields"]].values, props["data"])
    assert layers[0]["encoding"]["x"]["field"]["repeat"] == props["x_repeat"]
    assert layers[0]["encoding"]["y"]["field"]["repeat"] == props["y_repeat"]
    assert chart["repeat"]["column"] == props["x_repeat_fields"]
    assert chart["repeat"]["row"] == props["y_repeat_fields"]
    assert layers[1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data[props["label_field"]], props["label"])
    validate_matrix_plot_general_properties(chart["spec"], props)


def test_emb_axis_with_str_axes_metric(embset):
    p = embset.plot_interactive_matrix(
        embset["red"], embset["green"], embset["blue"], axes_metric="euclidean"
    )
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append(
            [
                scipy_distance.euclidean(e.vector, embset["red"].vector),
                scipy_distance.euclidean(e.vector, embset["green"].vector),
                scipy_distance.euclidean(e.vector, embset["blue"].vector),
            ]
        )
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data": vectors,
        "x_repeat": "column",
        "x_repeat_fields": ["red", "green", "blue"],
        "y_repeat": "row",
        "y_repeat_fields": ["blue", "green", "red"],
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "height": 200,
        "width": 200,
    }
    layers = chart["spec"]["layer"]
    chart_data = pd.DataFrame(chart["datasets"][chart["spec"]["data"]["name"]])
    assert layers[0]["mark"] == props["type"]
    assert np.array_equal(chart_data[props["x_repeat_fields"]].values, props["data"])
    assert layers[0]["encoding"]["x"]["field"]["repeat"] == props["x_repeat"]
    assert layers[0]["encoding"]["y"]["field"]["repeat"] == props["y_repeat"]
    assert chart["repeat"]["column"] == props["x_repeat_fields"]
    assert chart["repeat"]["row"] == props["y_repeat_fields"]
    assert layers[1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data[props["label_field"]], props["label"])
    validate_matrix_plot_general_properties(chart["spec"], props)


def test_emb_axis_with_different_axes_metric(embset):
    p = embset.plot_interactive_matrix(
        embset["red"],
        embset["green"],
        embset["blue"],
        axes_metric=[None, "cosine_distance", np.dot],
    )
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append(
            [
                e > embset["red"],
                scipy_distance.cosine(e.vector, embset["green"].vector),
                np.dot(e.vector, embset["blue"].vector),
            ]
        )
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data": vectors,
        "x_repeat": "column",
        "x_repeat_fields": ["red", "green", "blue"],
        "y_repeat": "row",
        "y_repeat_fields": ["blue", "green", "red"],
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "height": 200,
        "width": 200,
    }
    layers = chart["spec"]["layer"]
    chart_data = pd.DataFrame(chart["datasets"][chart["spec"]["data"]["name"]])
    assert layers[0]["mark"] == props["type"]
    assert np.array_equal(chart_data[props["x_repeat_fields"]].values, props["data"])
    assert layers[0]["encoding"]["x"]["field"]["repeat"] == props["x_repeat"]
    assert layers[0]["encoding"]["y"]["field"]["repeat"] == props["y_repeat"]
    assert chart["repeat"]["column"] == props["x_repeat_fields"]
    assert chart["repeat"]["row"] == props["y_repeat_fields"]
    assert layers[1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data[props["label_field"]], props["label"])
    validate_matrix_plot_general_properties(chart["spec"], props)


def test_mixed_axis(embset):
    p = embset.plot_interactive_matrix("blue", 2, embset["red"])
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append([e > embset["blue"], e.vector[2], e > embset["red"]])
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data": vectors,
        "x_repeat": "column",
        "x_repeat_fields": ["blue", "Dimension 2", "red"],
        "y_repeat": "row",
        "y_repeat_fields": ["red", "Dimension 2", "blue"],
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "height": 200,
        "width": 200,
    }
    layers = chart["spec"]["layer"]
    chart_data = pd.DataFrame(chart["datasets"][chart["spec"]["data"]["name"]])
    assert layers[0]["mark"] == props["type"]
    assert np.array_equal(chart_data[props["x_repeat_fields"]].values, props["data"])
    assert layers[0]["encoding"]["x"]["field"]["repeat"] == props["x_repeat"]
    assert layers[0]["encoding"]["y"]["field"]["repeat"] == props["y_repeat"]
    assert chart["repeat"]["column"] == props["x_repeat_fields"]
    assert chart["repeat"]["row"] == props["y_repeat_fields"]
    assert layers[1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data[props["label_field"]], props["label"])
    validate_matrix_plot_general_properties(chart["spec"], props)


def test_mixed_axis_with_str_axes_metric(embset):
    p = embset.plot_interactive_matrix(
        "blue", 2, embset["red"], axes_metric="cosine_similarity"
    )
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append(
            [
                1 - scipy_distance.cosine(e.vector, embset["blue"].vector),
                e.vector[2],
                1 - scipy_distance.cosine(e.vector, embset["red"].vector),
            ]
        )
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data": vectors,
        "x_repeat": "column",
        "x_repeat_fields": ["blue", "Dimension 2", "red"],
        "y_repeat": "row",
        "y_repeat_fields": ["red", "Dimension 2", "blue"],
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "height": 200,
        "width": 200,
    }
    layers = chart["spec"]["layer"]
    chart_data = pd.DataFrame(chart["datasets"][chart["spec"]["data"]["name"]])
    assert layers[0]["mark"] == props["type"]
    assert np.array_equal(chart_data[props["x_repeat_fields"]].values, props["data"])
    assert layers[0]["encoding"]["x"]["field"]["repeat"] == props["x_repeat"]
    assert layers[0]["encoding"]["y"]["field"]["repeat"] == props["y_repeat"]
    assert chart["repeat"]["column"] == props["x_repeat_fields"]
    assert chart["repeat"]["row"] == props["y_repeat_fields"]
    assert layers[1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data[props["label_field"]], props["label"])
    validate_matrix_plot_general_properties(chart["spec"], props)


def test_mixed_axis_with_different_axes_metric(embset):
    p = embset.plot_interactive_matrix(
        "blue", embset["red"], 3, axes_metric=[np.dot, None, "cosine_distance"]
    )
    chart = json.loads(p.to_json())
    vectors = []
    for e in embset.embeddings.values():
        vectors.append(
            [
                np.dot(e.vector, embset["blue"].vector),
                e > embset["red"],
                e.vector[3],
            ]
        )
    vectors = np.array(vectors)
    props = {
        "type": "circle",
        "data": vectors,
        "x_repeat": "column",
        "x_repeat_fields": ["blue", "red", "Dimension 3"],
        "y_repeat": "row",
        "y_repeat_fields": ["Dimension 3", "red", "blue"],
        "label_field": "original",
        "label": [v.orig for v in embset.embeddings.values()],
        "height": 200,
        "width": 200,
    }
    layers = chart["spec"]["layer"]
    chart_data = pd.DataFrame(chart["datasets"][chart["spec"]["data"]["name"]])
    assert layers[0]["mark"] == props["type"]
    assert np.array_equal(chart_data[props["x_repeat_fields"]].values, props["data"])
    assert layers[0]["encoding"]["x"]["field"]["repeat"] == props["x_repeat"]
    assert layers[0]["encoding"]["y"]["field"]["repeat"] == props["y_repeat"]
    assert chart["repeat"]["column"] == props["x_repeat_fields"]
    assert chart["repeat"]["row"] == props["y_repeat_fields"]
    assert layers[1]["encoding"]["text"]["field"] == props["label_field"]
    assert np.array_equal(chart_data[props["label_field"]], props["label"])
    validate_matrix_plot_general_properties(chart["spec"], props)


def test_raises_error_when_number_of_axes_is_not_equal_to_number_of_metrics(embset):
    with pytest.raises(
        ValueError, match="The number of given axes metrics should be the same"
    ):
        embset.plot_interactive_matrix("red", 3, 1, axes_metric=["euclidean", np.dot])
