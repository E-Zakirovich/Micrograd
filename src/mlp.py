from layer import Layer


class MLP:
    def __init__(self, nin, nouts):
        size = [nin] + nouts
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(size) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        parameters = []
        for layer in self.layers:
            for neurons in layer.parameters():
                parameters.append(neurons)

        return parameters