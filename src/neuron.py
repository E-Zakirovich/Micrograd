from value import Value
from random import uniform


class Neuron:
    def __init__(self, nin):
        self.weight = [Value(uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.weight, x)), self.bias)
        result = act.tanh()
        return result

    def parameters(self):
        return self.weight + [self.bias] 