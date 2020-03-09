from operator import add, sub, rshift, or_
from whatlies.common import plot_graph_layout



class EmbeddingSet:
    """
    This object represents a set of `Embedding`s. You can use the same operations
    as an `Embedding` but here we apply it to the entire set instead of a single
    `Embedding`.
    """
    def __init__(self, *embeddings, operations=None):
        if len(embeddings) == 1:
            # we assume it is a dictionary here
            self.embeddings = embeddings[0]
        else:
            # we assume it is a tuple of tokens
            self.embeddings = {t.name: t for t in embeddings}
        self.operations = [] if not operations else operations

    def operate(self, other, operation):
        """
        Attaches an operation to perform on the `EmbeddingSet`.

        **Inputs**

        - other: the other `Embedding`
        - operation: the operation to apply to all embeddings in the set, can be `+`, `-`, `|`, `>>`, `>`

        **Output**

        A new `EmbeddingSet`
        """
        new_embeddings = {k: operation(emb, other) for k, emb in self.embeddings.items()}
        return EmbeddingSet(new_embeddings, operations=self.operations + [(other, operation)])

    def __add__(self, other):
        return self.operate(other, add)

    def __sub__(self, other):
        return self.operate(other, sub)

    def __or__(self, other):
        return self.operate(other, or_)

    def __rshift__(self, other):
        return self.operate(other, rshift)

    def __getitem__(self, thing):
        return self.embeddings[thing]

    def __repr__(self):
        result = "EmbSet"
        translator = {add: '+', sub: '-', or_: '|', rshift: '>>'}
        for tok, op in self.operations:
            result = f"({result} {translator[op]} {tok.name})"
        return result

    def plot(self, kind="scatter", x_axis=None, y_axis=None, color=None, show_operations=False, **kwargs):
        """
        Handles the logic to perform a 2d plot in matplotlib.

        **Input**

        - kind: what kind of plot to make, can be `scatter`, `arrow` or `text`
        - x_axis: what embedding to use as a x-axis
        - y_axis: what embedding to us as a y-axis
        - color: the color to apply, only works for `scatter` and `arrow`
        - xlabel: manually override the xlabel
        - ylabel: manually override the ylabel
        - show_operations: setting to also show the applied operations, only works for `text`
        """
        for k, token in self.embeddings.items():
            token.plot(kind=kind, x_axis=x_axis, y_axis=y_axis, color=color, show_operations=show_operations, **kwargs)
        return self

    def plot_graph_layout(self, kind='cosine', **kwargs):
        """
        Handles the logic to plot a 2d graph using cosine distance
        :return:
        """
        plot_graph_layout(self.embeddings, kind, **kwargs)
        return self