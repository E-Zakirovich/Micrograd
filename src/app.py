from mlp import MLP
from value import Value

data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 0.0),
    ([1.0, 0.0], 0.0),
    ([1.0, 1.0], 1.0),
]

# Initialize model
mlp = MLP(2, [4, 4, 1])

# Training loop
epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    total_loss = 0.0

    for x, target in data:
        # Forward pass
        pred = mlp(x)
        if isinstance(pred, list):  # handle list output
            pred = pred[0]

        # Loss (mean squared error)
        loss = (pred - Value(target)) ** 2
        total_loss += loss.data

        # Backward pass
        mlp.zero_grad = [p.grad for p in mlp.parameters()]  # reset grads
        loss.backward()

        # Gradient descent update
        for p in mlp.parameters():
            p.data -= learning_rate * p.grad
            p.grad = 0.0


print(round(mlp([1.0, 1.0]).data))