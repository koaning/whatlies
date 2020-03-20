import numpy as np
from whatlies.common import handle_2d_plot


class Embedding:
    """
    This object represents a word embedding. It contains a vector and a name.

    Arguments:
        name: the name of this embedding, includes operations
        vector: the numerical representation of the embedding
        orig: original name of embedding, is left alone

    Usage:

    ```
    from whatlies.embedding import Embedding

    foo = Embedding("foo", [0.1, 0.3])
    bar = Embedding("bar", [0.7, 0.2])

    foo | bar
    foo - bar + bar
    ```
    """

    def __init__(self, name, vector, orig=None):
        self.orig = name if not orig else orig
        self.name = name
        self.vector = np.array(vector)

    def __add__(self, other):
        """
        Add two embeddings together.

        Usage:

        ```
        from whatlies.embedding import Embedding

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])

        foo + bar
        ```
        """
        return self.__class__(
            name=f"({self.name} + {other.name})",
            vector=self.vector + other.vector,
            orig=self.orig,
        )

    def __sub__(self, other):
        """
        Subtract two embeddings.

        Usage:

        ```
        from whatlies.embedding import Embedding

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])

        foo - bar
        ```
        """
        return self.__class__(
            name=f"({self.name} - {other.name})",
            vector=self.vector - other.vector,
            orig=self.orig,
        )

    def __gt__(self, other):
        """
        Measures the size of one embedding to another one.

        Usage:

        ```
        from whatlies.embedding import Embedding

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])

        foo > bar
        ```
        """
        return (self.vector.dot(other.vector)) / (other.vector.dot(other.vector))

    def __rshift__(self, other):
        """
        Maps an embedding unto another one.

        Usage:

        ```
        from whatlies.embedding import Embedding

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])

        foo >> bar
        ```
        """
        new_vec = (
            (self.vector.dot(other.vector))
            / (other.vector.dot(other.vector))
            * other.vector
        )
        return self.__class__(
            name=f"({self.name} >> {other.name})", vector=new_vec, orig=self.orig
        )

    def __or__(self, other):
        """
        Makes one embedding orthogonal to the other one.

        Usage:

        ```
        from whatlies.embedding import Embedding

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])

        foo | bar
        ```
        """
        new_vec = self.vector - (self >> other).vector
        return self.__class__(
            name=f"({self.name} | {other.name})", vector=new_vec, orig=self.orig
        )

    def __repr__(self):
        return f"Emb[{self.name}]"

    def plot(
        self,
        kind: str = "scatter",
        x_axis: str = None,
        y_axis: str = None,
        color: str = None,
        show_ops: bool = False,
        annot: bool = False,
    ):
        """
        Handles the logic to perform a 2d plot in matplotlib.

        Arguments:
            kind: what kind of plot to make, can be `scatter`, `arrow` or `text`
            x_axis: the x-axis to be used, must be given when dim > 2
            x_axis: the y-axis to be used, must be given when dim > 2
            color: the color of the dots
            show_ops: setting to also show the applied operations, only works for `text`
            annot: should the points be annotated

        **Usage**
        ```python
        from whatlies.embedding import Embedding

        foo = Embedding("foo", [0.1, 0.3])
        bar = Embedding("bar", [0.7, 0.2])

        foo.plot(kind="arrow", annot=True)
        bar.plot(kind="arrow", annot=True)
        ```
        """
        if len(self.vector) == 2:
            handle_2d_plot(
                self,
                kind=kind,
                color=color,
                show_operations=show_ops,
                xlabel=x_axis,
                ylabel=y_axis,
                annot=annot,
            )
            return self
        x_val = self > x_axis
        y_val = self > y_axis
        intermediate = Embedding(name=self.name, vector=[x_val, y_val], orig=self.orig)
        handle_2d_plot(
            intermediate,
            kind=kind,
            color=color,
            xlabel=x_axis.name,
            ylabel=y_axis.name,
            show_operations=show_ops,
            annot=annot,
        )
        return self
