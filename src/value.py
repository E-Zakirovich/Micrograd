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
        other = other if isinstance(other, Value) else Value(other)
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
        other = other if isinstance(other, Value) else Value(other)
        return self * other

    def __neg__(self):
        return self * (-1.0)
