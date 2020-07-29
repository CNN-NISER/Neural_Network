import numpy as np

from network import NeuralNetwork

# Create instance of NeuralNetwork
model = NeuralNetwork()

# Input data
inp_num = []
exp = []
for i in range(1,1000):
    inp_num.append(i%5)
    exp.append(3*(i%5))
test = [2,3,4]

model.layer(4)
model.layer(5)
model.layer(1)  #Output layer
model.training(inp_num, exp, 10, 0.1, 0.1)
print(model.getOutput(test))