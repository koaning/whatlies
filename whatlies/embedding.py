import numpy as np
from whatlies.common import handle_2d_plot


class Embedding:
    """
    This object represents a word embedding.

    **Inputs**

    - name: the name of the embedding
    - vector: the numeric encoding of the embedding
    - orig: the original name of the original embedding, is handled automatically
    """
    def __init__(self, name, vector, orig=None):
        self.orig = name if not orig else orig
        self.name = name
        self.vector = np.array(vector)

    def __add__(self, other):
        return self.__class__(name=f"({self.name} + {other.name})",
                              vector=self.vector + other.vector,
                              orig=self.orig)

    def __sub__(self, other):
        return self.__class__(name=f"({self.name} - {other.name})",
                              vector=self.vector - other.vector,
                              orig=self.orig)

    def __gt__(self, other):
        return (self.vector.dot(other.vector)) / (other.vector.dot(other.vector))

    def __rshift__(self, other):
        new_vec = (self.vector.dot(other.vector)) / (other.vector.dot(other.vector)) * other.vector
        return self.__class__(name=f"({self.name} >> {other.name})", vector=new_vec, orig=self.orig)

    def __or__(self, other):
        new_vec = self.vector - (self >> other).vector
        return self.__class__(name=f"({self.name} | {other.name})", vector=new_vec, orig=self.orig)

    def __repr__(self):
        return f"Emb[{self.name}]"

    def plot(self, kind="scatter", x_axis=None, y_axis=None, color=None, show_operations=False,
             annot=False):
        """
        Handles the logic to perform a 2d plot in matplotlib.

        **Input**

        - kind: what kind of plot to make, can be `scatter`, `arrow` or `text`
        - color: the color to apply, only works for `scatter` and `arrow`
        - xlabel: manually override the xlabel
        - ylabel: manually override the ylabel
        - show_operations: setting to also show the applied operations, only works for `text`
        """
        if len(self.vector) == 2:
            handle_2d_plot(self, kind=kind, color=color, show_operations=show_operations,
                           xlabel=x_axis.name, ylabel=y_axis.name, annot=annot)
        else:
            x_val = self > x_axis
            y_val = self > y_axis
            intermediate = Embedding(name=self.name, vector=[x_val, y_val], orig=self.orig)
            handle_2d_plot(intermediate, kind=kind, color=color,
                           xlabel=x_axis.name, ylabel=y_axis.name, show_operations=show_operations, annot=annot)
        return self
