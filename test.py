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

model.addInput(x) # Input layer
model.layer(4) # Hidden layer 1
model.layer(8) # Hidden layer 2
model.layer(16) # Hidden layer 3
model.layer(1) # Output layer

# Train the model
model.train(y, 200)

# Get predictions for test data
predictions = model.predict(test)

# Print out the results
for i in range(len(predictions)):
    print(f"{test[i]} * 2 = {predictions[i]}")