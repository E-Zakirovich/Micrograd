from mlp import MLP

x = [1.0, 3.0, 2.0]
n = MLP(
    nin=3,
    nouts=[3, 3, 1]
)
o = n(x)
print(o)