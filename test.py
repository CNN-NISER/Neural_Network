import numpy as np

from network import NeuralNetwork

# Input data
x = np.random.randn(3, 1) # Random input vector of three numbers (3x1)

# Create instance of NeuralNetwork
model = NeuralNetwork()

model.inputLayer(x)
model.layer(4)
model.layer(5)
model.layer(2)

print(model.getOutput())