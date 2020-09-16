def validate_plot_general_properties(ax, props):
    assert ax.xaxis.get_label_text() == props["x_label"]
    assert ax.yaxis.get_label_text() == props["y_label"]
    assert ax.get_title() == props["title"]
    assert ax.get_aspect() == props["aspect"]
