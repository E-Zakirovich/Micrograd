from math import exp


class Value:
    def __init__(self, data, _children=(), _operation="", label=""):
        self.data = data
        self._prev = set(_children)
        self._operation = _operation
        self.label = label

    def __repr__(self):
        return f"Value(data = {self.data} | type = {type(self.data)})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data=self.data + other.data,
            _children=(self, other),
            _operation=" add ",
        )
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data=self.data * other.data,
            _children=(self, other),
            _operation=" mul ",
        )
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * (-1.0)

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Power must be int or float"
        out = Value(
            data=self.data ** other,
            _children=(self, other),
            _operation=f" {self.label} ^ {other} ",
        )
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** (-1.0)

    def __rtruediv__(self, other):
        return other * self ** (-1.0)

    def exp(self):
        out = Value(
            data=exp(self.data),
            _children=(self,),
            _operation=" exp ",
        )
        return out

    def tanh(self):
        x = exp(self.data * 2.0)
        out = Value(
            data=(x - 1) / (x + 1),
            _children=(self,),
            _operation=" tan ",
        )
        return out