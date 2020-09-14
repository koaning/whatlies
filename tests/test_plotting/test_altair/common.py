def validate_plot_general_properties(chart_layer, props):
    assert chart_layer["encoding"]["x"]["axis"]["title"] == props["x_label"]
    assert chart_layer["encoding"]["y"]["axis"]["title"] == props["y_label"]
    assert chart_layer["title"] == props["title"]
    assert chart_layer["mark"]["type"] == props["type"]


def validate_matrix_plot_general_properties(spec, props):
    assert spec["width"] == props["width"]
    assert spec["height"] == props["height"]
