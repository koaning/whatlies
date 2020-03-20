import matplotlib.pylab as plt
import networkx as nx

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import distance_metrics


def handle_2d_plot(embedding, kind, color=None, xlabel=None, ylabel=None, show_operations=False,
                   annot=False):
    """
    Handles the logic to perform a 2d plot in matplotlib.

    **Input**

    - embedding.md: a `whatlies.Embedding` object to plot
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
    if (kind == "text") or annot:
        plt.text(embedding.vector[0] + 0.01, embedding.vector[1], name)

    plt.xlabel("x" if not xlabel else xlabel)
    plt.ylabel("y" if not ylabel else ylabel)


def plot_graph_layout(embedding_set, kind='cosine', **kwargs):
    """
    Handles the plotting of a layout graph using the embeddings in an embeddingset as input.

    **Input**

    - embeddings: a set of `whatlies.Embedding` objects to plot
    - kind: distance metric options: 'cityblock', 'cosine', 'euclidean', 'l2', 'l1', 'manhattan',
    """

    vectors = [token.vector for k, token in embedding_set.items()]
    label_dict = {i: w for i, (w, _) in enumerate(embedding_set.items())}
    dist_fnc = distance_metrics()[kind]
    dist = dist_fnc(np.array(vectors), np.array(vectors))
    # Greate graph
    graph = nx.from_numpy_matrix(dist)
    distance = pd.DataFrame(dist).to_dict()
    # Chhange layout positions of the graph
    pos = nx.kamada_kawai_layout(graph, dist=distance)
    # Draw nodes and labels
    nx.draw_networkx_nodes(graph, pos, node_color='b', alpha=0.5)
    nx.draw_networkx_labels(graph, pos, labels=label_dict, **kwargs)
