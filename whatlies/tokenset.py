from operator import add, sub, rshift, or_


class TokenSet:
    """
    An embedding object that you can play with.
    """

    def __init__(self, *tokens, operations=None):
        if len(tokens) == 1:
            # we assume it is a dictionary here
            self.tokens = tokens[0]
        else:
            # we assume it is a tuple of tokens
            self.tokens = {t.name: t for t in tokens}
        self.operations = [] if not operations else operations

    def operate(self, other, operation):
        return TokenSet(self.tokens, operations=self.operations + [(other, operation)])

    def __add__(self, other):
        return self.operate(other, add)

    def __sub__(self, other):
        return self.operate(other, sub)

    def __or__(self, other):
        return self.operate(other, or_)

    def __rshift__(self, other):
        return self.operate(other, rshift)

    def __getitem__(self, thing):
        return self.tokens[thing]

    def __repr__(self):
        result = "TokenSet"
        translator = {add: '+', sub: '-', or_: '|', rshift: '>>'}
        for tok, op in self.operations:
            result = f"({result} {translator[op]} {tok.name})"
        return result

    def plot(self, kind="scatter", x_axis=None, y_axis=None, color=None, show_operations=False, **kwargs):
        for k, token in self.tokens.items():
            token.plot(kind=kind, x_axis=x_axis, y_axis=y_axis, color=color, show_operations=show_operations, **kwargs)
        return self
