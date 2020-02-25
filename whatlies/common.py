import matplotlib.pylab as plt


def handle_2d_plot(embedding, kind, color=None, xlabel=None, ylabel=None, show_operations=False):
    """
    Handles the logic to perform a 2d plot in matplotlib.

    **Input**

    - embedding: a `whatlies.Embedding` object to plot
    - kind: what kind of plot to make, can be `scatter`, `arrow` or `text`
    - color: the color to apply, only works for `scatter` and `arrow`
    - xlabel: manually override the xlabel
    - ylabel: manually override the ylabel
    - show_operations: setting to also show the applied operations, only works for `text`
    """
    name = embedding.name if show_operations else embedding.orig
    if kind == "scatter":
        if color is None:
            color = "steelblue"
        plt.scatter([embedding.vector[0]], [embedding.vector[1]], c=color)
    if kind == "arrow":
        plt.scatter([embedding.vector[0]], [embedding.vector[1]], c="white", s=0.01)
        plt.quiver([0], [0], [embedding.vector[0]], [embedding.vector[1]], color=color,
                   angles='xy', scale_units='xy', scale=1)
        plt.text(embedding.vector[0] + 0.01, embedding.vector[1], name)
    if kind == "text":
        plt.text(embedding.vector[0] + 0.01, embedding.vector[1], name)
    plt.xlabel("x" if not xlabel else xlabel)
    plt.ylabel("y" if not ylabel else ylabel)
