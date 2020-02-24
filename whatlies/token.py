import numpy as np
from whatlies.common import handle_2d_plot


class Token:
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
        """
        return the scale when we map self unto other
        """
        return (self.vector.dot(other.vector)) / (other.vector.dot(other.vector))

    def __rshift__(self, other):
        new_vec = (self.vector.dot(other.vector)) / (other.vector.dot(other.vector)) * other.vector
        return self.__class__(name=f"({self.name} >> {other.name})", vector=new_vec, orig=self.orig)

    def __or__(self, other):
        new_vec = self.vector - (self >> other).vector
        return self.__class__(name=f"({self.name} | {other.name})", vector=new_vec, orig=self.orig)

    def __repr__(self):
        return f"Token[{self.name}]"

    def plot(self, kind="scatter", x_axis=None, y_axis=None, color=None, show_operations=False):
        """
        kind must be in [scatter, arrow, text]
        """
        if len(self.vector) == 2:
            handle_2d_plot(self, kind=kind, color=color)
        else:
            x_val = self > x_axis
            y_val = self > y_axis
            intermediate = Token(name=self.name, vector=[x_val, y_val], orig=self.orig)
            handle_2d_plot(intermediate, kind=kind, color=color,
                           xlabel=x_axis.name, ylabel=y_axis.name, show_operations=show_operations)
        return self
