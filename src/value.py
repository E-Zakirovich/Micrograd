class Value:
    def __init__(self, data, _children=(), _operation="", label=""):
        self.data = data
        self._prev = _children
        self._operation = _operation
        self.label = label

    def __repr__(self):
        return f"Value(data = {self.data} | type = {type(self.data)})"