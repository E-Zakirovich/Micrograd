from value import Value
from neuron import Neuron

x = [1.0, 3.0, 2.0]
n = Neuron(3)

o = n(x)
print(o)