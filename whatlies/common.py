import matplotlib.pylab as plt


def handle_2d_plot(token, kind, color=None):
    """
    Generate a 2d plot of a token.
    :param token:
    :param kind:
    :param color:
    :return:
    """
    if kind == "scatter":
        if color is None:
            color = "steelblue"
        plt.scatter([token.vector[0]], [token.vector[1]], c=color)
    if kind == "arrow":
        plt.scatter([token.vector[0]], [token.vector[1]], c="white", s=0.01)
        plt.quiver([0], [0], [token.vector[0]], [token.vector[1]], color=color,
                   angles='xy', scale_units='xy', scale=1)
        plt.text(token.vector[0] + 0.01, token.vector[1], token.name)
    if kind == "text":
        plt.text(token.vector[0] + 0.01, token.vector[1], token.name)
    plt.xlabel("x")
    plt.ylabel("y")
