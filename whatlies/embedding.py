import numpy as np
import matplotlib.pylab as plt
from operator import add, sub, rshift, or_


class Embedding:
    """
    An embedding object that you can play with.
    """

    def __init__(self, language, operations=None):
        self.language = language
        if not operations:
            operations = None
        self.operations = operations

    def operate(self, other, operation):
        return Embedding(language=self.language,
                         operations=self.operations + [(other, operation)])

    def __add__(self, other):
        return self.operate(other, add)

    def __sub__(self, other):
        return self.operate(other, sub)

    def __or__(self, other):
        return self.operate(other, or_)

    def __rshift__(self, other):
        return self.operate(other, rshift)

    def __getitem__(self, thing):
        if isinstance(self.language, dict):
            return self.language[thing]

    def __repr__(self):
        result = "Embedding"
        translator = {add: '+', sub: '-', or_: '|', rshift: '>>'}
        for tok, op in self.operations:
            result = f"({result} {translator[op]} {tok.name})"
        return result