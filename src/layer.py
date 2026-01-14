from value import Value
from neuron import Neuron

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self):
        params = []
        for neuron in self.neurons:
            for element in neuron.parameters():
                params.append(element)

        return params