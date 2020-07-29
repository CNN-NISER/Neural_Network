import numpy as np

from network import NeuralNetwork

# Training data (0 to 9)
x = np.arange(10).reshape(10, 1)
y = 2 * x

# Test data (10 to 15)
test = np.arange(10, 16)
test = test.reshape(len(test), 1)

# Create instance of NeuralNetwork
model = NeuralNetwork()
# Linear classifier (no hidden layers)
model.addInput(x)
model.layer(1)

model.train(y, 200)

result = model.predict(test)

for i in range(len(result)):
    print(f"{test[i]} * 2 = {result[i]}")