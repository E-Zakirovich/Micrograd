from math import exp


class Value:
    def __init__(self, data, _children=(), _operation="", label=""):
        self.data = data
        self._prev = set(_children)
        self._operation = _operation
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data = {self.data} | type = {type(self.data)})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            data=self.data + other.data,
            _children=(self, other),
            _operation=" add ",
        )

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * other.grad

        out._backward = _backward

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

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * other.grad

        out._backward = _backward

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

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = exp(self.data * 2.0)
        out = Value(
            data=(x - 1) / (x + 1),
            _children=(self,),
            _operation=" tan ",
        )

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out