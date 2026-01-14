from value import Value
from layer import Layer
x = [1.0, 3.0, 2.0]
n = Layer(3, 3)
o = n(x)
print(o)